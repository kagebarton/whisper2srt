#!/usr/bin/env python3
"""Standalone test script for stem-based FFmpeg mixing with ZMQ volume control.

Usage:
    python test_stems.py <vocal_file> <nonvocal_file>

Plays audio in realtime through PulseAudio. While playing, type a volume
value (0.0-1.0) and press Enter to adjust vocal volume via ZMQ.
"""

import subprocess
import sys
import threading


def send_zmq_volume(volume: float) -> bool:
    try:
        import zmq
    except ImportError:
        print("pyzmq not installed")
        return False

    try:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.connect("tcp://127.0.0.1:5556")
        command = f"volume@vocalvol volume {volume}"
        sock.send_string(command)
        reply = sock.recv_string()
        sock.close()
        ctx.term()
        print(f"ZMQ reply: {reply}")
        return True
    except Exception as e:
        print(f"ZMQ error: {e}")
        return False


def run_ffmpeg(vocal, nonvocal):
    filter_complex = (
        "[0:a]volume@vocalvol=1.0,azmq=bind_address=tcp\\\\://127.0.0.1\\\\:5556[vocal];"
        "[1:a]volume@nonvocalvol=1.0[nonvocal];"
        "[vocal][nonvocal]amix=inputs=2:normalize=0[out]"
    )

    cmd = [
        "ffmpeg",
        "-i", vocal,
        "-i", nonvocal,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-f", "pulse",
        "-name", "stem_test",
        "default",
    ]

    print("Running:", " ".join(cmd))
    print()
    return subprocess.Popen(cmd, stderr=subprocess.PIPE)


def drain_stderr(proc):
    for line in iter(proc.stderr.readline, b""):
        print("[ffmpeg]", line.decode("utf-8", "ignore").strip())


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    vocal = sys.argv[1]
    nonvocal = sys.argv[2]

    proc = run_ffmpeg(vocal, nonvocal)

    stderr_thread = threading.Thread(target=drain_stderr, args=(proc,), daemon=True)
    stderr_thread.start()

    print("Playing. Enter a volume (0.0-1.0) to adjust vocal via ZMQ, or 'q' to quit:")
    try:
        while proc.poll() is None:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if line == "q":
                proc.terminate()
                break
            try:
                vol = float(line)
                send_zmq_volume(vol)
            except ValueError:
                print("Enter a float or 'q'")
    except KeyboardInterrupt:
        proc.terminate()

    proc.wait()
    print(f"FFmpeg exited with code {proc.returncode}")


if __name__ == "__main__":
    main()
