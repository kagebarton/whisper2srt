#!/usr/bin/env python3
"""
Async FFmpeg mixer that combines two audio files using a filter_complex graph.
Launches FFmpeg as a separate process asynchronously.

Usage:
    python mixer.py vocal.m4a nonvocal.m4a
    python mixer.py vocal.m4a nonvocal.m4a --keychange 2
    python mixer.py vocal.m4a nonvocal.m4a --output alsa_output.pci-0000_00_1b.0.analog-stereo

    Change vocal volume during playback:
    
    python3 -c "
    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.connect('tcp://127.0.0.1:5556')
    sock.send_string('volume@vocalvol volume 0.2')
    print(sock.recv())
    "

Options:
    vocal               Path to vocal audio file (required)
    nonvocal            Path to non-vocal audio file (required)
    --keychange         Pitch shift in semitones (e.g., +2 for up 2 semitones, -3 for down 3)
    --output            PulseAudio output device (default: default)

Example:
    # Mix vocal and non-vocal tracks with default volumes
    python mixer.py song/vocal.m4a song/nonvocal.m4a

    # Pitch shift up 2 semitones
    python mixer.py song/vocal.m4a song/nonvocal.m4a --keychange 2
"""

import asyncio
import argparse
import sys


def build_filter_complex(keychange: float | None = None) -> str:
    """
    Build the filter_complex string piece by piece.

    Args:
        keychange: Pitch shift in semitones (default: None, no pitch shift)

    Returns:
        Complete filter_complex string for FFmpeg
    """
    filters = []

    # Calculate pitch multiplier if keychange is specified
    # pitch = 2^(semitones/12)
    if keychange is not None:
        pitch = 2 ** (keychange / 12)

        vocal_rb = f"rubberband=pitch={pitch}"
        f":window=long"
        f":pitchq=quality"
        f":transients=crisp"
        f":detector=compound"
        f":formant=preserved,"
        
        nonvocal_rb = f"rubberband=pitch={pitch}"
        f":window=standard"
        f":pitchq=consistency"
        f":transients=crisp"
        f":detector=compound"
        f":formant=shifted,"
    else:
        vocal_rb = None
        nonvocal_rb = None

    # Vocal input: apply volume, optional pitch shift, and azmq for runtime control
    vocal_chain = f"[0:a]volume@vocalvol=1.0"
    if vocal_rb:
        vocal_chain += f",{vocal_rb}"
    vocal_chain += ",azmq=bind_address=tcp\\\\://127.0.0.1\\\\:5556[vocal]"
    filters.append(vocal_chain)

    # Non-vocal input: apply volume and optional pitch shift
    nonvocal_chain = f"[1:a]volume@nonvocalvol=1.0"
    if nonvocal_rb:
        nonvocal_chain += f",{nonvocal_rb}"
    nonvocal_chain += "[nonvocal]"
    filters.append(nonvocal_chain)

    # Mix both streams
    mix_filter = "[vocal][nonvocal]amix=inputs=2:normalize=0[out]"
    filters.append(mix_filter)

    # Join all filters with semicolons
    filter_complex = ";".join(filters)

    return filter_complex


async def run_ffmpeg(vocal_file: str, nonvocal_file: str,
                     keychange: float | None = None,
                     output: str = "default") -> asyncio.subprocess.Process:
    """
    Launch FFmpeg asynchronously to mix two audio files.

    Args:
        vocal_file: Path to vocal audio file
        nonvocal_file: Path to non-vocal audio file
        keychange: Pitch shift in semitones
        output: PulseAudio output device (default: "default")

    Returns:
        asyncio.subprocess.Process object
    """
    filter_complex = build_filter_complex(keychange)

    cmd = [
        "ffmpeg",
        "-re",
        "-i", vocal_file,
        "-i", nonvocal_file,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-f", "pulse",
        output
    ]

    print(f"Launching FFmpeg with filter_complex:")
    print(f"  {filter_complex}")
    print()

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    return process


async def monitor_process(process: asyncio.subprocess.Process) -> int:
    """
    Monitor the FFmpeg process and stream its output.

    Args:
        process: The FFmpeg subprocess to monitor

    Returns:
        Exit code of the process
    """
    # Stream stderr (FFmpeg outputs progress info there)
    while True:
        line = await process.stderr.readline()
        if not line:
            break
        decoded = line.decode().rstrip()
        if decoded:
            print(decoded)

    # Wait for process to complete
    return await process.wait()


async def main():
    parser = argparse.ArgumentParser(
        description="Async FFmpeg audio mixer for vocal and non-vocal tracks"
    )
    parser.add_argument("vocal", help="Path to vocal audio file (e.g., vocal.m4a)")
    parser.add_argument("nonvocal", help="Path to non-vocal audio file (e.g., nonvocal.m4a)")
    parser.add_argument("--keychange", type=float, default=None,
                        help="Pitch shift in semitones (e.g., 2 for up 2 semitones, -3 for down 3)")
    parser.add_argument("--output", default="default",
                        help="PulseAudio output device (default: default)")

    args = parser.parse_args()

    # Validate input files exist
    import os
    if not os.path.isfile(args.vocal):
        print(f"Error: Vocal file not found: {args.vocal}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.nonvocal):
        print(f"Error: Non-vocal file not found: {args.nonvocal}", file=sys.stderr)
        sys.exit(1)

    # Launch FFmpeg asynchronously
    process = await run_ffmpeg(
        args.vocal,
        args.nonvocal,
        args.keychange,
        args.output
    )

    print(f"FFmpeg process started with PID: {process.pid}")
    print("Press Ctrl+C to stop\n")

    # Monitor and wait for completion
    try:
        exit_code = await monitor_process(process)
        print(f"\nFFmpeg exited with code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nStopping FFmpeg...")
        process.terminate()
        await process.wait()
        print("Stopped.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
