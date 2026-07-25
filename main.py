"""
astrbot_plugin_zvv —— 张维为表情包插件（ONNX 轻量版）

功能：
- AI 主动调用 search_zvv 搜索候选 → send_zvv 选图发送
- 手动 /zvv <描述> 直接发图
- ONNX Runtime 推理，零配置，开箱即用
"""

from __future__ import annotations

import os
import pickle
import time
import numpy as np
from pathlib import Path

# ── 全局线程限制（必须在导入 ONNX/tokenizers 之前设置）──
_cpu_count = os.cpu_count() or 2
_threads = str(max(1, int(_cpu_count * 0.5)))
os.environ.setdefault("OMP_NUM_THREADS", _threads)       # ONNX Runtime (OpenMP)
os.environ.setdefault("RAYON_NUM_THREADS", "1")           # tokenizers (Rust)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _threads)   # NumPy (OpenBLAS)
os.environ.setdefault("MKL_NUM_THREADS", _threads)        # NumPy (MKL)

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, StarTools


# ═══════════════════════════════════════
#  ONNX 嵌入引擎（替代 PyTorch）
# ═══════════════════════════════════════

class EmbeddingEngine:
    """ONNX Runtime 嵌入引擎。

    使用场景：
    1. 部署前运行一次 python export_model.py，生成 model_zvv/
    2. 插件加载 model_zvv/ 下的 ONNX 模型和 tokenizer
    3. 以后不再需要 PyTorch
    """

    def __init__(
        self,
        model_dir: str,
        num_threads: int = 2,
        max_length: int = 512,
    ):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.max_length = max_length

        # ── 加载 tokenizer ──
        tok_path = os.path.join(model_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(
                f"tokenizer.json 不存在: {tok_path}\n"
                f"请确保模型目录包含 tokenizer.json"
            )
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=max_length)
        self.tokenizer.enable_truncation(max_length=max_length)

        # ── 加载 ONNX 模型（自动检测文件名）──
        for candidate in ("model_q8.onnx", "model_fp16.onnx", "model.onnx"):
            onnx_path = os.path.join(model_dir, candidate)
            if os.path.exists(onnx_path):
                break
        else:
            raise FileNotFoundError(
                f"未找到 ONNX 模型文件: {model_dir}"
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 低内存模式：双核服务器友好
        opts.enable_cpu_mem_arena = False

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._hidden_dim = self.session.get_outputs()[0].shape[-1]
        logger.info(
            f"[ZVV] ONNX 引擎就绪: {num_threads} threads, "
            f"hidden_dim={self._hidden_dim}, model={onnx_path}"
        )

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """编码文本为归一化向量。

        Returns:
            (batch_size, hidden_dim) 归一化向量，可直接做点积 = 余弦相似度
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        # ── tokenize ──
        encodings = self.tokenizer.encode_batch(texts)
        batch_size = len(encodings)
        seq_len = max(len(e.ids) for e in encodings)

        input_ids = np.zeros((batch_size, seq_len), dtype=np.int64)
        attention_mask = np.zeros((batch_size, seq_len), dtype=np.int64)

        for i, enc in enumerate(encodings):
            n = len(enc.ids)
            input_ids[i, :n] = enc.ids
            attention_mask[i, :n] = enc.attention_mask

        # ── ONNX 推理 → sentence_embedding（模型已完成 mean pooling）──
        outputs = self.session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        sentence_embeds = outputs[1]  # sentence_embedding（index 1），已归一化
        return sentence_embeds[0] if single else sentence_embeds


# ═══════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════

def _unwrap_event(event: AstrMessageEvent) -> AstrMessageEvent:
    """兼容 v4.26+ ContextWrapper。"""
    try:
        from astrbot.core.agent.run_context import ContextWrapper
        if isinstance(event, ContextWrapper):
            return event.context.event
    except ImportError:
        pass
    return event


async def _send_qq_sticker(
    event: AstrMessageEvent,
    file_path: str,
    summary: str = "[动画表情]",
) -> bool:
    """QQ 平台以动画表情格式发送。"""
    try:
        from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
            AiocqhttpMessageEvent,
        )
    except ImportError:
        return False
    if not isinstance(event, AiocqhttpMessageEvent):
        return False
    try:
        chain = MessageChain(chain=[Image(file=file_path)])
        obmsg = await event._parse_onebot_json(chain)
        obmsg[0]["data"]["summary"] = summary
        await event.bot.send(event.message_obj.raw_message, obmsg)
        return True
    except Exception:
        return False


# ═══════════════════════════════════════
#  主插件
# ═══════════════════════════════════════

class Main(Star):
    """张维为表情包插件 —— AI 两步选图发送（ONNX 轻量版）。"""

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)

        # ── 持久化路径（重装插件不丢数据）──
        data_root = Path(StarTools.get_data_dir("astrbot_plugin_zvv"))
        self.image_dir = data_root / "images"
        self.model_dir = str(data_root / "model_zvv")
        self.cache_dir = data_root / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "embeddings.pkl"

        # ── 线程数（env var 已全局限制，这里仅记录日志）──
        self._num_threads = max(1, int(_cpu_count * 0.5))
        self._top_k = 20

        # ── 运行时状态 ──
        self._engine: EmbeddingEngine | None = None
        self._embeddings: np.ndarray | None = None
        self._image_names: list[str] = []
        self._turn_candidates: dict[str, tuple[float, list[dict]]] = {}
        self._candidate_max_age = 300  # 5 分钟自动过期

    # ══════════════════════════════
    #  嵌入引擎 & 缓存
    # ══════════════════════════════

    def _get_engine(self) -> EmbeddingEngine:
        if self._engine is None:
            self._engine = EmbeddingEngine(
                self.model_dir,
                num_threads=self._num_threads,
            )
        return self._engine

    def _load_cache(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self._embeddings = np.array(data["embeddings"], dtype=np.float32)
            self._image_names = list(data["names"])
            missing = [n for n in self._image_names if not (self.image_dir / n).exists()]
            if missing:
                logger.warning(f"[ZVV] {len(missing)} 张图片已不存在，重建缓存")
                return False
            logger.info(f"[ZVV] 加载缓存: {len(self._image_names)} 张图, {self._embeddings.shape}")
            return True
        except Exception as e:
            logger.warning(f"[ZVV] 缓存加载失败: {e}")
            return False

    def _build_cache(self) -> bool:
        if not self.image_dir.exists():
            logger.error(f"[ZVV] 图片目录不存在: {self.image_dir}")
            return False
        exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        files = sorted(
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        if not files:
            logger.error(f"[ZVV] 图片目录为空: {self.image_dir}")
            return False

        engine = self._get_engine()
        batch_size = 32
        names = []
        all_vecs = []
        logger.info(f"[ZVV] 预计算 {len(files)} 张图向量...")

        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            texts = [f.stem for f in batch]
            batch_vecs = engine.encode(texts)
            names.extend(f.name for f in batch)
            all_vecs.append(batch_vecs)
            if len(all_vecs) * batch_size % 128 < batch_size:
                logger.info(f"[ZVV] 进度: {min(i + batch_size, len(files))}/{len(files)}")
            # 每批让出 CPU，避免卡死 AstrBot
            time.sleep(0.05)

        embeddings = np.concatenate(all_vecs, axis=0).astype(np.float32)

        self._embeddings = embeddings
        self._image_names = names

        with open(self.cache_path, "wb") as f:
            pickle.dump({"embeddings": embeddings, "names": names}, f)

        logger.info(f"[ZVV] 缓存构建完成: {len(names)} 张图, {embeddings.shape}")
        return True

    def _ensure_ready(self) -> bool:
        if self._embeddings is not None:
            return True
        if self._load_cache():
            return True
        return self._build_cache()

    # ══════════════════════════════
    #  语义搜索
    # ══════════════════════════════

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        if top_k is None:
            top_k = self._top_k
        if not self._ensure_ready():
            return []

        engine = self._get_engine()
        query_vec = engine.encode(query)
        scores = np.dot(self._embeddings, query_vec)
        # 先取 top_k * 3 高分候选，再用 MMR 挑出多样化的 top_k
        pool_size = min(top_k * 3, len(self._image_names))
        pool_indices = np.argsort(scores)[::-1][:pool_size]
        diverse = self._mmr_select(pool_indices, scores, top_k)
        return [
            {"id": idx + 1, "name": self._image_names[int(i)], "score": float(scores[int(i)])}
            for idx, i in enumerate(diverse)
        ]

    def _mmr_select(self, candidates: np.ndarray, scores: np.ndarray, top_k: int, lam: float = 0.6) -> list:
        """MMR 多样性重排：相关但不重复。lam 越大越看重相关度，越小越看重多样性。"""
        remaining = list(candidates)
        selected = [remaining.pop(0)]  # 第一个选最高分
        while len(selected) < top_k and remaining:
            mmr = []
            for r in remaining:
                rel = float(scores[r])
                sim_max = max(float(np.dot(self._embeddings[r], self._embeddings[s])) for s in selected)
                mmr.append(lam * rel - (1 - lam) * sim_max)
            best = remaining.pop(int(np.argmax(mmr)))
            selected.append(best)
        return selected

    # ══════════════════════════════
    #  会话状态
    # ══════════════════════════════

    def _get_session_key(self, event: AstrMessageEvent) -> str:
        event = _unwrap_event(event)
        try:
            sid = event.get_session_id()
            if sid:
                return str(sid)
        except Exception:
            pass
        try:
            return str(event.unified_msg_origin)
        except Exception:
            return "global"

    def _cleanup_stale(self) -> int:
        now = time.time()
        stale = [
            k for k, (ts, _) in self._turn_candidates.items()
            if now - ts > self._candidate_max_age
        ]
        for k in stale:
            del self._turn_candidates[k]
        return len(stale)

    def _get_candidates(self, event: AstrMessageEvent) -> list[dict]:
        self._cleanup_stale()
        _, candidates = self._turn_candidates.get(self._get_session_key(event), (0, []))
        return candidates

    def _set_candidates(self, event: AstrMessageEvent, candidates: list[dict]) -> None:
        self._cleanup_stale()
        self._turn_candidates[self._get_session_key(event)] = (time.time(), candidates)

    # ═══════════════════════════════════════
    #  LLM Tools
    # ═══════════════════════════════════════

    @filter.llm_tool(name="search_zvv")
    async def search_zvv(self, event: AstrMessageEvent, mood: str):
        """搜索张维为表情包候选。**先发完文字回复，再调用此工具选图发送。**

        Args:
            mood(string): 用简短的场景+情绪描述你要表达的梗。
                好的例子: "印度被0比7剃头" "美国加关税自讨苦吃" "日本军国主义可笑"
                不要只写情绪词如"自信""不屑"，结合具体话题才有区分度。

        选择原则：优先挑最乐子、最贻笑大方、最似绷非绷的那张，避开正经说教的。像"国际笑话""挺搞笑的""笑掉大牙"这类优先。 "表示赞同"
        """
        event = _unwrap_event(event)
        mood = str(mood or "").strip()
        if not mood:
            yield "[search_zvv] 请描述想表达的情绪，如：'对方太自信了'。"
            return

        logger.info(f"[ZVV] AI 搜索: {mood[:50]}")
        results = self.search(mood)
        if not results:
            yield f"[search_zvv] 没找到匹配「{mood}」的表情包，换个说法？"
            return

        self._set_candidates(event, results)
        lines = [f"[search_zvv] 找到 {len(results)} 张候选：\n"]
        for r in results:
            lines.append(f"  [{r['id']}] {Path(r['name']).stem}")
        lines.append("\n选一张调用 send_zvv(编号)。")
        yield "\n".join(lines)

    @filter.llm_tool(name="send_zvv")
    async def send_zvv(self, event: AstrMessageEvent, emoji_id: int):
        """发送选中的张维为表情包。必须先调用 search_zvv 获取候选，选好后再调用本工具发送。

        Args:
            emoji_id(number): 编号，从 search_zvv 返回的列表中选。
        """
        event = _unwrap_event(event)
        candidates = self._get_candidates(event)
        if not candidates:
            yield "[send_zvv] 没有候选，请先 search_zvv。"
            return

        try:
            emoji_id = int(emoji_id)
        except (TypeError, ValueError):
            yield f"[send_zvv] 编号「{emoji_id}」无效。"
            return
        if emoji_id < 1 or emoji_id > len(candidates):
            yield f"[send_zvv] 编号 {emoji_id} 超范围 (1-{len(candidates)})。"
            return

        selected = candidates[emoji_id - 1]
        path = self.image_dir / selected["name"]
        if not path.exists():
            yield f"[send_zvv] 文件丢失: {selected['name']}"
            return

        logger.info(f"[ZVV] AI 发送: {selected['name']}")
        sent = await _send_qq_sticker(event, str(path))
        if not sent:
            await event.send(MessageChain([Image(file=str(path))]))

        yield f"[send_zvv] 已发送: {Path(selected['name']).stem}"

    # ═══════════════════════════════════════
    #  手动命令
    # ═══════════════════════════════════════

    @filter.command("zvv")
    async def zvv_command(self, event: AstrMessageEvent, *, text: str = ""):
        """手动搜索发送: /zvv <描述>"""
        text = (text or "").strip()
        if not text:
            yield event.plain_result("用法: /zvv <描述>\n例: /zvv 无语")
            return

        results = self.search(text, top_k=1)
        if not results:
            yield event.plain_result(f"没找到「{text}」")
            return

        path = self.image_dir / results[0]["name"]
        logger.info(f"[ZVV] 手动: /zvv {text} → {results[0]['name']}")
        sent = await _send_qq_sticker(event, str(path))
        if not sent:
            yield event.chain_result([Image(file=str(path))])

    # ═══════════════════════════════════════
    #  生命周期
    # ═══════════════════════════════════════

    async def initialize(self):
        await super().initialize()

        # 检查模型是否存在
        onnx_ok = any(
            (Path(self.model_dir) / c).exists()
            for c in ("model_q8.onnx", "model_fp16.onnx", "model.onnx")
        )
        if not onnx_ok or not self.image_dir.exists() or not any(self.image_dir.iterdir()):
            logger.info("[ZVV] 模型或图片缺失，自动初始化（期间可能卡顿，仅此一次）...")
            logger.info("[ZVV] 模型缺失，自动运行 init.py 下载...")
            try:
                from .init import check_and_download
                if not check_and_download():
                    logger.error("[ZVV] 初始化失败，请手动运行 python init.py")
                    return
            except Exception as e:
                logger.error(f"[ZVV] 自动初始化失败: {e}，请手动运行 python init.py")
                return

        try:
            if not self._ensure_ready():
                logger.error("[ZVV] 缓存初始化失败")
                return
            self._get_engine()
            logger.info("[ZVV] 插件就绪（ONNX 引擎已预热）")
        except Exception as e:
            logger.error(f"[ZVV] 初始化异常: {e}")

    async def terminate(self):
        self._engine = None
        self._embeddings = None
        self._turn_candidates.clear()
        await super().terminate()
