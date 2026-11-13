import json
import os
import struct
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

INPUT_SIZE = 64+1280*720*3          # header + BGR color frame
OUTPUT_SIZE = 64+720*1280*4+65536   # header + RGBA projector image + JSON state
IMAGE_END = 64+720*1280*4


class ExternalTracker:
    """Spawns an isolated FFv2 color tracker process. Frames are submitted as
    top-down BGR; results come back as top-down uint8 RGBA + JSON state."""

    def __init__(self, camera):
        self.input = shared_memory.SharedMemory(create=True, size=INPUT_SIZE)
        self.output = shared_memory.SharedMemory(create=True, size=OUTPUT_SIZE)
        self.input.buf[:64] = bytes(64)
        self.output.buf[:64] = bytes(64)
        self.sequence = 0
        self.last_result = 0
        executable = os.environ.get('FF_PYTHON', 'C:/Python314/python.exe')
        self.process = subprocess.Popen([executable, str(Path(__file__).resolve()),
                                         self.input.name, self.output.name,
                                         camera, str(os.getpid())],
                                        creationflags=subprocess.CREATE_NO_WINDOW)

    def submit(self, bgr, timestamp):
        if self.process.poll() is not None:
            raise RuntimeError(f'FFv2 tracking worker exited: {self.process.returncode}')
        self.sequence += 2
        struct.pack_into('<Q', self.input.buf, 0, self.sequence-1)
        h, w = bgr.shape[:2]
        struct.pack_into('<IId', self.input.buf, 8, h, w, timestamp)
        np.ndarray((h, w, 3), np.uint8, buffer=self.input.buf, offset=64)[:] = bgr
        struct.pack_into('<Q', self.input.buf, 0, self.sequence)

    def poll(self):
        sequence = struct.unpack_from('<Q', self.output.buf, 0)[0]
        if not sequence or sequence % 2 or sequence == self.last_result:
            return None
        length = struct.unpack_from('<I', self.output.buf, 8)[0]
        image = np.ndarray((720, 1280, 4), np.uint8, buffer=self.output.buf, offset=64).copy()
        text = bytes(self.output.buf[IMAGE_END:IMAGE_END+length])
        if struct.unpack_from('<Q', self.output.buf, 0)[0] != sequence:
            return None
        self.last_result = sequence
        return image, json.loads(text)

    def shutdown(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.input.close()
        self.output.close()
        self.input.unlink()
        self.output.unlink()


def run(input_name, output_name, camera, parent_pid):
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    parent = kernel.OpenProcess(0x100000, False, int(parent_pid))
    if not parent:
        raise RuntimeError('Cannot watch TouchDesigner parent process')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ffv2_tracker
    tracker = ffv2_tracker.FFv2Tracker(camera)
    source = shared_memory.SharedMemory(name=input_name, track=False)
    destination = shared_memory.SharedMemory(name=output_name, track=False)
    last_sequence = 0
    result_sequence = 0
    last_frame = None
    last_timestamp = 0
    last_processing = 0
    while kernel.WaitForSingleObject(parent, 0) == 258:
        sequence = struct.unpack_from('<Q', source.buf, 0)[0]
        if sequence and not sequence % 2 and sequence != last_sequence:
            h, w, timestamp = struct.unpack_from('<IId', source.buf, 8)
            frame = np.ndarray((h, w, 3), np.uint8, buffer=source.buf, offset=64).copy()
            if struct.unpack_from('<Q', source.buf, 0)[0] != sequence:
                continue
            last_sequence = sequence
            last_frame, last_timestamp = frame, timestamp
        elif time.perf_counter()-last_processing < 0.1 or last_sequence == 0:
            time.sleep(.002)
            continue
        else:
            last_frame = None
        started = time.perf_counter()
        image, state = tracker.process_frame(last_frame,
                                             last_timestamp if last_frame is not None else started)
        last_processing = time.perf_counter()
        state['performance'] = dict(measurement_timestamp=last_timestamp,
                                    worker_ms=(last_processing-started)*1000,
                                    frame_age_ms=(last_processing-last_timestamp)*1000,
                                    process_id=os.getpid())
        text = json.dumps(state).encode()
        if len(text) > 65536:
            raise RuntimeError('Tracking state exceeds shared memory capacity')
        result_sequence += 2
        struct.pack_into('<Q', destination.buf, 0, result_sequence-1)
        np.ndarray((720, 1280, 4), np.uint8, buffer=destination.buf, offset=64)[:] = image
        destination.buf[IMAGE_END:IMAGE_END+len(text)] = text
        struct.pack_into('<I', destination.buf, 8, len(text))
        struct.pack_into('<Q', destination.buf, 0, result_sequence)
    source.close()
    destination.close()
    kernel.CloseHandle(parent)


if __name__ == '__main__':
    run(*sys.argv[1:])
