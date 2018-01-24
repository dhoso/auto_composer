import midi
import os
import numpy as np
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, help = "the number of bars", required=False, default=8)
parser.add_argument("-o", type=str, help = "output filename", required=False, default='output.mid')

command_arguments = parser.parse_args()

class Note():
    def __init__(self, pitch, position, length):
        self.pitch = pitch
        self.position = position
        self.length = length

    def __repr__(self):
        return "<Note pitch={}, position={}, length={}>".format(self.pitch, self.position, self.length)

chord_pitch_pattern = {
    'C': [0, 4, 7],
    'D': [2, 6, 9],
    'E': [4, 8, 11],
    'F': [5, 9, 0],
    'G': [7, 11, 2],
    'A': [9, 1, 4],
    'B': [11, 3, 6],
    'Cm': [0, 3, 7],
    'Dm': [2, 5, 9],
    'Em': [4, 7, 11],
    'Fm': [5, 8, 0],
    'Gm': [7, 10, 2],
    'Am': [9, 0, 4],
    'Bm': [11, 2, 6]
}

chord_base = {
    'C': 48,
    'D': 50,
    'E': 52,
    'F': 53,
    'G': 55,
    'A': 57,
    'B': 59
}

def get_chords(pattern):
    pattern.make_ticks_abs()
    track = pattern[1]
    ticks = sorted(list(set(list(map(lambda event: event.tick, track)))))

    chords = []
    for tick in ticks:
        events = list(filter(lambda event: event.tick == tick
                                            and isinstance(event, midi.NoteOnEvent)
                                            and event.velocity != 0,
                             track))
        chord = get_chord_by_events(events)
        bar = tick / pattern.resolution
        chords.append([bar, chord])
    return chords

def get_chord_by_events(events):
    scales = list(map(lambda event: event.pitch % 12, events))
    chord_cands = list(filter(lambda c: set(chord_pitch_pattern[c]) == set(scales), chord_pitch_pattern))
    if len(chord_cands) == 1:
        return chord_cands[0]
    else:
        return None

def get_melody(pattern):
    pattern.make_ticks_abs()
    track = pattern[1]
    on_events = list(filter(lambda event: isinstance(event, midi.NoteOnEvent)
                                          and event.velocity != 0,
                            track))
    off_events = list(filter(lambda event: isinstance(event, midi.NoteOffEvent)
                                           or (isinstance(event, midi.NoteOnEvent)
                                               and event.velocity == 0),
                            track))

    notes = []
    for event in on_events:
        off_event = min(list(filter(lambda off_e: off_e.tick > event.tick, off_events)), key=lambda x: x.tick)
        note_length = (off_event.tick - event.tick) / pattern.resolution
        position = event.tick / pattern.resolution
        notes.append(Note(event.pitch, position, note_length))
    return notes

def build_chord_graph(chords_lists):
    chord_graph = {}

    for chords in chords_lists:
        pre_chord = None
        for chord in chords:
            if pre_chord not in chord_graph:
                chord_graph[pre_chord] = {}

            if chord[1] not in chord_graph[pre_chord]:
                chord_graph[pre_chord][chord[1]] = 0

            chord_graph[pre_chord][chord[1]] += 1

            pre_chord = chord[1]
    return chord_graph

def compose_chords(chord_graph, first_chord=None, num = 8):
    beat = 4
    chords = []
    current_chord = first_chord
    for i in range(num):
        chord_graph_cur_chord = { k: v for k, v in chord_graph[current_chord].items() if k != None}
        items = list(chord_graph_cur_chord.items())
        next_chord_cands = list(map(lambda x: x[0], items))
        weights = list(map(lambda x: x[1], items))
        probability = np.array(weights) / np.sum(weights)
        next_chord = np.random.choice(next_chord_cands, 1, p=probability)[0]
        chords.append([i * beat ,next_chord])
        current_chord = next_chord
    return chords

def extract_rhythm(melody):
    beat = 4

    def group_notes_by_bar(notes):
        all_note_group = []
        note_group = []
        current_bar = 0
        for n in notes:
            if n.position >= current_bar * beat + beat:
                all_note_group.append(note_group)
                note_group = []
                current_bar += 1
            note_group.append(n)
        all_note_group.append(note_group)
        return all_note_group

    def get_position_in_bar(notes):
        return tuple(map(lambda x: x.position % beat, notes))

    grouped_notes = group_notes_by_bar(melody)
    rhythm = list(map(lambda x: get_position_in_bar(x), grouped_notes))
    return rhythm

def build_rhythm_graph(rhythm_list, rhythm_graph={}):
    pre_rhythm = None
    for rhythm in rhythm_list:
        if pre_rhythm not in rhythm_graph:
            rhythm_graph[pre_rhythm] = {}

        if rhythm not in rhythm_graph[pre_rhythm]:
            rhythm_graph[pre_rhythm][rhythm] = 0

        rhythm_graph[pre_rhythm][rhythm] += 1

        pre_rhythm = rhythm
    return rhythm_graph

def compose_rhythm(rhythm_graph, num = 8):
    beat = 4

    rhythm_seq = []
    current_rhythm = None
    for i in range(num):
        rhythm_graph_cur_rhythm = { k: v for k, v in rhythm_graph[current_rhythm].items() if k != None}
        items = list(rhythm_graph_cur_rhythm.items())
        next_rhythm_cands = list(map(lambda x: x[0], items))
        weights = list(map(lambda x: x[1], items))
        probability = np.array(weights) / np.sum(weights)

        # converts rhythm_cands to their indices because np.random.choice cannot treat a tuple as one object
        next_rhythm_cands_indices =[i for i, v in enumerate(next_rhythm_cands)]
        next_rhyth_index = np.random.choice(next_rhythm_cands_indices, 1, p=probability)[0]
        next_rhythm = next_rhythm_cands[next_rhyth_index]

        rhythm_seq.append(next_rhythm_cands[next_rhyth_index])
        current_rhythm = next_rhythm

    return [note + bar * beat for bar, rhythm in enumerate(rhythm_seq) for note in rhythm]

def build_melody_graph(melody, chords, melody_graph={}):
    pre_pitch = None
    for note in melody:
        chord = get_chord_at_position(note.position, chords)

        key = (pre_pitch, chord)
        if key not in melody_graph:
            melody_graph[key] = {}

        if note.pitch not in melody_graph[key]:
            melody_graph[key][note.pitch] = 0

        melody_graph[key][note.pitch] += 1

        pre_pitch = note.pitch
    return melody_graph

def compose_melody(melody_graph, rhythm, chords):
    def get_melody_graph_by_chord(melody_graph, chord):
        melody_graph_selected_by_chord = {k: v for k, v in melody_graph.items() if k[1] == chord }
        melody_graph_cur_key = {}
        for _, val_dict in melody_graph_selected_by_chord.items():
            for k, v in val_dict.items():
                if k not in melody_graph_cur_key:
                    melody_graph_cur_key[k] = v
                else:
                    melody_graph_cur_key[k] += v
        return melody_graph_cur_key

    melody = []
    current_pitch = None
    for note_pos in rhythm:
        current_chord = get_chord_at_position(note_pos, chords)
        key = (current_pitch, current_chord)
        if key in melody_graph:
            melody_graph_cur_key = { k: v for k, v in melody_graph[key].items() if k != None}
        else:
            melody_graph_cur_key = get_melody_graph_by_chord(melody_graph, current_chord)

        items = list(melody_graph_cur_key.items())
        next_pitch_cands = list(map(lambda x: x[0], items))
        weights = list(map(lambda x: x[1], items))
        probability = np.array(weights) / np.sum(weights)
        next_pitch = np.random.choice(next_pitch_cands, 1, p=probability)[0]
        melody.append(Note(next_pitch, note_pos, None))
        current_pitch = next_pitch
    return melody

def get_chord_at_position(position, chords):
    return max(list(filter(lambda c: c[0] <= position, chords)), key=lambda c:c[0])[1]

def write_midifile(melody, chords, pattern, filename):
    beat = 4

    def get_chord_pitches(chord):
        base = chord_base[chord[0]]
        if 'm' in chord:
            return (base, base + 3, base + 7)
        else:
            return (base, base + 4, base + 7)

    pattern = copy.deepcopy(pattern)
    del pattern[1]
    track_melody = midi.Track()
    pattern.append(track_melody)

    track_melody.append(midi.ProgramChangeEvent(tick=0, value=0, channel=0))
    track_melody.append(midi.TrackNameEvent(tick=0, text='Track 1#', data=[84, 114, 97, 99, 107, 32, 49]))

    pre_note = None
    for note in melody:
        if pre_note != None:
            tick = int((note.position - pre_note.position) * pattern.resolution)
            track_melody.append(midi.NoteOffEvent(tick=tick, pitch=pre_note.pitch, channel=0))
        track_melody.append(midi.NoteOnEvent(tick=0, velocity=100, pitch=note.pitch, channel=0))
        pre_note = note
    track_melody.append(midi.NoteOffEvent(tick=pattern.resolution * 1, pitch=pre_note.pitch, channel=0))
    track_melody.append(midi.EndOfTrackEvent(tick=0, channel=0))

    track_chord = midi.Track()
    pattern.append(track_chord)
    track_chord.append(midi.ProgramChangeEvent(tick=0, value=0, channel=1))
    track_chord.append(midi.TrackNameEvent(tick=0, text='Track 2#', data=[84, 114, 97, 99, 107, 32, 49], channel=1))
    pre_chord = None
    for chord in chords:
        if pre_chord != None:
            tick = int((chord[0] - pre_chord[0]) * pattern.resolution)
            track_chord.append(midi.NoteOnEvent(tick=tick, velocity=0, pitch=0, channel=1))

            for pitch in get_chord_pitches(pre_chord[1]):
                track_chord.append(midi.NoteOnEvent(tick=0, velocity=0, pitch=pitch, channel=1))

        for pitch in get_chord_pitches(chord[1]):
            track_chord.append(midi.NoteOnEvent(tick=0, velocity=80, pitch=pitch, channel=1))
        pre_chord = chord
    for i, pitch in enumerate(get_chord_pitches(pre_chord[1])):
        track_chord.append(midi.NoteOnEvent(tick=pattern.resolution * beat if i == 0 else 0, velocity=0, pitch=pitch, channel=1))
    track_chord.append(midi.EndOfTrackEvent(tick=0, channel=1))

    midi.write_midifile(filename, pattern)

def build_graphs(melody_pattern, chord_pattern):
    melody = get_melody(melody_pattern)
    rhythm = extract_rhythm(melody)
    chords = get_chords(chord_pattern)

    rhythm_graph = build_rhythm_graph(rhythm)
    melody_graph = build_melody_graph(melody, chords)
    chord_graph = build_chord_graph([chords])

    return [melody_graph, rhythm_graph, chord_graph]

# graphs: [melody_graph, rhythm_graph, chord_graph]
def compose_music(graphs, num=8):
    melody_graph, rhythm_graph, chord_graph = graphs
    chords = compose_chords(chord_graph, num=num)
    rhythm = compose_rhythm(rhythm_graph, num=num)
    melody = compose_melody(melody_graph, rhythm, chords)

    return melody, chords

filepath = os.path.join('data', '3_melo.mid')
melody_pattern = midi.read_midifile(filepath)
filepath = os.path.join('data', '3_chord.mid')
chord_pattern = midi.read_midifile(filepath)
graphs = build_graphs(melody_pattern, chord_pattern)
melody, chords = compose_music(graphs, command_arguments.n)
write_midifile(melody, chords, melody_pattern, command_arguments.o)
