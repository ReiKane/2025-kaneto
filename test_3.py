import csv
from collections import defaultdict

def load_press_csv(csv_path):
    """
    戻り値:
    key_times[key] = [time_sec, time_sec, ...]
    """
    key_times = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["time_sec"])
            k = int(row["key_index"])
            key_times[k].append(t)

    return key_times

def times_to_intervals(times, frame_dt, gap_frames=1):
    """
    times: [t0, t1, t2, ...]
    frame_dt: 1 / fps
    """
    intervals = []

    start = times[0]
    prev = times[0]

    for t in times[1:]:
        # 連続フレームか？
        if t - prev <= frame_dt * (gap_frames + 0.5):
            prev = t
        else:
            intervals.append((start, prev + frame_dt))
            start = t
            prev = t

    intervals.append((start, prev + frame_dt))
    return intervals

BASE_MIDI_NOTE = 60  # C4

def key_to_midi_note(key):
    return BASE_MIDI_NOTE + key

def build_events_from_csv(csv_path, fps):
    key_times = load_press_csv(csv_path)
    frame_dt = 1.0 / fps

    events = []  # (time_sec, type, midi_note)
    MIN_DURATION = 0.1

    for key, times in key_times.items():
        times.sort()
        intervals = times_to_intervals(times, frame_dt)

        for start, end in intervals:
            duration = end - start
            if duration < MIN_DURATION:
                continue
            note = key_to_midi_note(key)
            events.append((start, "on", note))
            events.append((end, "off", note))

    # 時刻順に並べる（超重要）
    events.sort(key=lambda x: x[0])
    return events

from mido import Message, MidiFile, MidiTrack

def write_midi(events, output_path="output.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message("program_change", program=0, time=0))

    last_time = 0.0
    ticks_per_second = mid.ticks_per_beat * 2  # tempo=120 BPM想定

    for t, etype, note in events:
        delta_sec = t - last_time
        delta_ticks = int(delta_sec * ticks_per_second)

        if etype == "on":
            msg = Message("note_on", note=note, velocity=64, time=delta_ticks)
        else:
            msg = Message("note_off", note=note, velocity=64, time=delta_ticks)

        track.append(msg)
        last_time = t

    mid.save(output_path)

fps = 60  # 動画と同じFPS
events = build_events_from_csv("press_log.csv", fps)
write_midi(events, "from_csv_1.mid")
