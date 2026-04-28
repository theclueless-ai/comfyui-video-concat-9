import os
import tempfile
import subprocess
import folder_paths


# ═══════════════════════════════════════════════════════════════════════════
# Carga robusta de VideoFromFile
# ═══════════════════════════════════════════════════════════════════════════

def _load_video_from_file_class():
    candidates = [
        "comfy_api.latest._input_impl.video_types",
        "comfy_api.input_impl.video_types",
        "comfy_api.input_impl",
        "comfy_api.latest.input_impl.video_types",
    ]
    for mod_path in candidates:
        try:
            mod = __import__(mod_path, fromlist=["VideoFromFile"])
            if hasattr(mod, "VideoFromFile"):
                return getattr(mod, "VideoFromFile")
        except Exception:
            continue
    return None


_VideoFromFile = _load_video_from_file_class()
if _VideoFromFile is not None:
    print(f"[VideoConcat9] VideoFromFile cargado: {_VideoFromFile}")
else:
    print("[VideoConcat9] ⚠️ VideoFromFile no encontrado, se usará wrapper propio")


class _FallbackVideo:
    def __init__(self, path):
        self._path = str(path)

    def get_path(self):
        return self._path

    def save_to(self, dest_path, *args, **kwargs):
        import shutil
        shutil.copy(self._path, dest_path)
        return dest_path

    def get_dimensions(self):
        try:
            import cv2
            cap = cv2.VideoCapture(self._path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return (w, h)
        except Exception:
            return (0, 0)

    def get_duration(self):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", self._path],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    @property
    def path(self):
        return self._path

    def __str__(self):
        return self._path

    def __repr__(self):
        return f"_FallbackVideo(path={self._path!r})"


def _wrap_as_video(path):
    if _VideoFromFile is not None:
        try:
            return _VideoFromFile(path)
        except Exception as e:
            print(f"[VideoConcat9] Error instanciando VideoFromFile: {e}, usando fallback")
    return _FallbackVideo(path)


def _is_blocker(value):
    """Detecta un ExecutionBlocker por nombre de clase o instancia."""
    if value is None:
        return False
    cls_name = type(value).__name__
    if cls_name == "ExecutionBlocker":
        return True
    try:
        from comfy_execution.graph import ExecutionBlocker
        if isinstance(value, ExecutionBlocker):
            return True
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Extracción de audio desde el mp4 concatenado
# ═══════════════════════════════════════════════════════════════════════════

def _silent_audio(sample_rate=44100, channels=2, samples=1):
    """Devuelve un AUDIO de ComfyUI con silencio. Útil cuando no hay pista."""
    import torch
    waveform = torch.zeros(1, channels, samples)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _has_audio_stream(video_path):
    """Comprueba con ffprobe si el archivo tiene stream de audio."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "a",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True
        )
        return "audio" in (result.stdout or "")
    except Exception:
        return False


def _extract_audio(video_path):
    """
    Extrae el track de audio del archivo de vídeo y lo devuelve como
    dict AUDIO de ComfyUI: {"waveform": Tensor[B,C,T], "sample_rate": int}.

    - Si el archivo no tiene audio, devuelve silencio (para no romper
      PreviewAudio o nodos que esperan AUDIO obligatorio).
    - Usa ffmpeg → wav temporal → torchaudio (con fallback a wave+numpy).
    """
    import torch

    if not _has_audio_stream(video_path):
        print(f"[VideoConcat9] ℹ️ El video concatenado no tiene stream de audio, se devuelve silencio")
        return _silent_audio()

    audio_tmp = tempfile.mktemp(suffix=".wav")
    try:
        # Extraemos a wav PCM 16-bit estéreo a 44.1k para máxima compatibilidad.
        # Si el original es mono, ffmpeg lo upmixa; si es 48k, lo resamplea.
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            audio_tmp
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(audio_tmp):
            print(f"[VideoConcat9] ⚠️ ffmpeg no pudo extraer audio: {result.stderr[-200:]}")
            return _silent_audio()

        # Carga: prioridad torchaudio (es lo que usa ComfyUI internamente),
        # fallback a wave+numpy.
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_tmp)
            # torchaudio devuelve [channels, samples], ComfyUI quiere [batch, channels, samples]
            waveform = waveform.unsqueeze(0)
            print(f"[VideoConcat9] 🔊 Audio extraído: {waveform.shape}, {sample_rate} Hz")
            return {"waveform": waveform, "sample_rate": int(sample_rate)}
        except Exception as e_ta:
            print(f"[VideoConcat9] torchaudio no disponible ({e_ta}), usando wave+numpy")
            import wave
            import numpy as np
            with wave.open(audio_tmp, 'rb') as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                n = wf.getnframes()
                raw = wf.readframes(n)
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                arr = arr.reshape(-1, ch).T  # [channels, samples]
            else:
                arr = arr.reshape(1, -1)     # [1, samples]
            waveform = torch.from_numpy(arr).unsqueeze(0)  # [1, channels, samples]
            print(f"[VideoConcat9] 🔊 Audio extraído (wave): {waveform.shape}, {sr} Hz")
            return {"waveform": waveform, "sample_rate": int(sr)}
    finally:
        try:
            if os.path.exists(audio_tmp):
                os.unlink(audio_tmp)
        except Exception:
            pass


def _make_silent_video(video_path, temp_dir):
    """
    Crea una copia del video sin stream de audio. Usa stream copy para vídeo
    (rapidísimo, sin recodificar) y omite el track de audio con -an.
    Si falla, devuelve la ruta original como fallback.
    """
    counter = 0
    while True:
        out_name = f"video_concat_silent_{counter:04d}.mp4"
        silent_path = os.path.join(temp_dir, out_name)
        if not os.path.exists(silent_path):
            break
        counter += 1

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "copy",
        "-an",
        silent_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(silent_path):
        print(f"[VideoConcat9] ⚠️ No se pudo crear video sin audio: {result.stderr[-200:]}")
        return video_path  # fallback al original
    print(f"[VideoConcat9] 🔇 Video sin audio creado: {silent_path}")
    return silent_path


# ═══════════════════════════════════════════════════════════════════════════
# VideoConcat9 — SIN lazy evaluation
# ═══════════════════════════════════════════════════════════════════════════
#
# IMPORTANTE: NO usamos lazy=True porque en el log anterior causaba que
# ComfyUI cancelase los Seedance posteriores al ExecutionBlocker.
#
# En ComfyUI, cuando un input OPCIONAL recibe un ExecutionBlocker, el
# comportamiento es que el nodo aún se ejecuta y puede detectar/filtrar
# ese valor. El blocker NO se propaga automáticamente al output si el
# input es optional y el nodo decide no usarlo.
#
# La clave está en que TODAS las 9 entradas son opcionales, por lo que
# ComfyUI pasará el blocker como valor del kwarg y nosotros lo filtramos.
# ═══════════════════════════════════════════════════════════════════════════

class VideoConcat9:
    """
    Concatena hasta 9 videos del tipo VIDEO de ComfyUI usando ffmpeg.

    - 9 entradas opcionales con lazy=True.
    - check_lazy_status pide los inputs UNO POR UNO para que los bloqueados
      por ExecutionBlocker no arrastren a los demás. Cuando recibe un
      ExecutionBlocker lo marca como visto y pide el siguiente.
    - Ramas bloqueadas por DurationGate se filtran.
    - El archivo resultante se escribe a temp/ y se entrega como VIDEO
      listo para Save Video.
    - Salida ADICIONAL: AUDIO extraído del archivo concatenado, para que
      no haga falta colgar un GetVideoComponents detrás (que falla con
      el wrapper de VIDEO de este nodo).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "video_1": ("VIDEO", {"lazy": True}),
                "video_2": ("VIDEO", {"lazy": True}),
                "video_3": ("VIDEO", {"lazy": True}),
                "video_4": ("VIDEO", {"lazy": True}),
                "video_5": ("VIDEO", {"lazy": True}),
                "video_6": ("VIDEO", {"lazy": True}),
                "video_7": ("VIDEO", {"lazy": True}),
                "video_8": ("VIDEO", {"lazy": True}),
                "video_9": ("VIDEO", {"lazy": True}),
                "force_reencode": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("VIDEO", "AUDIO", "VIDEO")
    RETURN_NAMES = ("video", "audio", "video_silent")
    FUNCTION = "concat_videos"
    CATEGORY = "video/utils"
    OUTPUT_NODE = False

    def check_lazy_status(self, **kwargs):
        """
        Estrategia: pedir los inputs uno por uno.

        En cada llamada, ComfyUI nos pasa los inputs ya evaluados hasta el
        momento. Si detectamos que hay algún input aún no evaluado (None)
        que está conectado en el grafo, pedimos solo UNO para evaluar.
        Así, si viene un ExecutionBlocker, solo afecta a ese input —
        los demás no se han evaluado aún y pueden seguir.

        NOTA IMPORTANTE: ComfyUI solo incluye en kwargs los inputs que
        tienen link en el grafo. Los no conectados ni aparecen.
        Un input "no evaluado aún" = está en kwargs con valor None.
        """
        for i in range(1, 10):
            key = f"video_{i}"
            if key in kwargs and kwargs[key] is None:
                # Solo pedimos UN input por llamada. ComfyUI llamará otra
                # vez a check_lazy_status después de evaluar este.
                return [key]
        # Todos evaluados (o bloqueados, que ya los tendremos en kwargs
        # como ExecutionBlocker): nada más que pedir.
        return []

    def _get_path(self, video):
        if isinstance(video, str):
            return video

        for attr in ("get_path", "path", "video_path", "filepath", "filename"):
            val = getattr(video, attr, None)
            if val is not None:
                if callable(val):
                    try:
                        result = val()
                        if result:
                            return str(result)
                    except Exception:
                        continue
                else:
                    if val:
                        p = str(val)
                        if not os.path.isabs(p):
                            output_dir = folder_paths.get_output_directory()
                            full = os.path.join(output_dir, p)
                            if os.path.exists(full):
                                return full
                        return p

        if hasattr(video, "save_to"):
            tmp = tempfile.mktemp(suffix=".mp4")
            try:
                video.save_to(tmp)
                if os.path.exists(tmp):
                    return tmp
            except Exception:
                pass

        if hasattr(video, "__dict__"):
            d = video.__dict__
            for key in ("path", "video_path", "filepath", "filename", "_path", "_file"):
                if key in d and d[key]:
                    return str(d[key])

        if isinstance(video, dict):
            for key in ("path", "video_path", "filename", "filepath", "url"):
                if key in video and video[key]:
                    p = str(video[key])
                    if not os.path.isabs(p):
                        output_dir = folder_paths.get_output_directory()
                        full = os.path.join(output_dir, p)
                        if os.path.exists(full):
                            return full
                    return p

        raise ValueError(
            f"No se pudo extraer la ruta del VIDEO. "
            f"Tipo: {type(video)}, __dict__: {getattr(video, '__dict__', 'N/A')}"
        )

    def _get_temp_dir(self):
        try:
            temp_dir = folder_paths.get_temp_directory()
        except Exception:
            temp_dir = os.path.join(folder_paths.get_output_directory(), "..", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir

    def concat_videos(self, **kwargs):
        force_reencode = kwargs.get("force_reencode", False)

        videos = []
        for i in range(1, 10):
            key = f"video_{i}"
            v = kwargs.get(key, None)

            if v is None:
                continue

            if _is_blocker(v):
                print(f"[VideoConcat9] ⏭️ {key} bloqueado por ExecutionBlocker, se ignora")
                continue

            try:
                path = self._get_path(v)
            except Exception as e:
                print(f"[VideoConcat9] ⚠️ {key} ignorado: {e}")
                continue

            if not path or not os.path.exists(path):
                print(f"[VideoConcat9] ⚠️ {key} no existe en disco, se salta: {path}")
                continue

            videos.append((i, path))
            print(f"[VideoConcat9] ✅ {key} válido: {path}")

        if len(videos) == 0:
            raise ValueError(
                "[VideoConcat9] No hay ningún video válido. "
                "O todas las entradas están vacías, o todas las ramas están bloqueadas."
            )

        if len(videos) == 1:
            idx, only_path = videos[0]
            print(f"[VideoConcat9] Solo video_{idx}, passthrough: {only_path}")
            audio_out = _extract_audio(only_path)
            silent_path = _make_silent_video(only_path, self._get_temp_dir())
            return (_wrap_as_video(only_path), audio_out, _wrap_as_video(silent_path))

        print(f"[VideoConcat9] Concatenando {len(videos)} vídeos en posiciones: {[i for i, _ in videos]}")

        temp_dir = self._get_temp_dir()
        counter = 0
        while True:
            out_name = f"video_concat_tmp_{counter:04d}.mp4"
            out_path = os.path.join(temp_dir, out_name)
            if not os.path.exists(out_path):
                break
            counter += 1

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for _, p in videos:
                safe_path = p.replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")
            list_path = f.name

        try:
            success = False
            if not force_reencode:
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", list_path,
                    "-c", "copy",
                    out_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    success = True
                else:
                    print(f"[VideoConcat9] copy falló, recodificando... {result.stderr[-200:]}")

            if not success:
                cmd_reencode = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", list_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    out_path
                ]
                result2 = subprocess.run(cmd_reencode, capture_output=True, text=True)
                if result2.returncode != 0:
                    raise RuntimeError(f"ffmpeg falló:\n{result2.stderr[-500:]}")
        finally:
            try:
                os.unlink(list_path)
            except Exception:
                pass

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"[VideoConcat9] El archivo resultante no se generó: {out_path}")

        print(f"[VideoConcat9] ✅ Concatenado: {out_path} ({os.path.getsize(out_path)} bytes)")

        # Extraemos el audio DEL ARCHIVO CONCATENADO en disco — es la fuente
        # de verdad, evitamos depender de get_components() del wrapper VIDEO.
        audio_out = _extract_audio(out_path)

        # Generamos versión sin audio para conectarla a AudioVideoCombine
        # sin riesgo de solape con la pista que vuelve de ElevenLabs.
        silent_path = _make_silent_video(out_path, temp_dir)

        video_obj = _wrap_as_video(out_path)
        silent_obj = _wrap_as_video(silent_path)
        print(f"[VideoConcat9] Tipo de salida: {type(video_obj).__name__}")
        return (video_obj, audio_out, silent_obj)


NODE_CLASS_MAPPINGS = {
    "VideoConcat9": VideoConcat9,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoConcat9": "🎬 Video Concat (9 videos)",
}
