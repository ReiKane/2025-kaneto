import csv
from collections import defaultdict

def load_press_csv(csv_path):
    notes = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            notes.append({
                "start": float(row["start_time"]),
                "end": float(row["end_time"]),
                "midi": int(row["midi_note"])
            })
    return notes

def build_events(notes):
    events = []
    for n in notes:
        events.append((n["start"], "on", n["midi"]))
        events.append((n["end"], "off", n["midi"]))
    events.sort(key=lambda x: x[0])
    return events
from mido import Message, MidiFile, MidiTrack

def write_midi(events, output_path="output.mid"):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message("program_change", program=0, time=0))

    last_time = 0.0
    ticks_per_second = mid.ticks_per_beat * 2  # 120 BPM

    for t, etype, note in events:
        delta_sec = t - last_time
        delta_ticks = int(delta_sec * ticks_per_second)

        msg = Message(
            "note_on" if etype == "on" else "note_off",
            note=note,
            velocity=64,
            time=delta_ticks
        )
        track.append(msg)
        last_time = t

    mid.save(output_path)
    print(mid.ticks_per_beat)

notes = load_press_csv("finger_log.csv")
events = build_events(notes)
write_midi(events, "from_finger_csv.mid")
