"""
PURPOSE: Las frases de las pruebas de clap_texto.py: formas de frase, paráfrasis, negación, idioma, conceptos no sonoros,
         longitud, instrumentos, artistas y el vocabulario a barrer. Solo datos, para poder leerlos y cambiarlos sin tocar
         las métricas. Las captions se escribieron describiendo el sonido, sin nombrar el género.
CHANGELOG:
  - 2026-11-09: Creación inicial.
"""

FORMAS = {
    "bare": "{}",
    "music": "{} music",
    "track": "a {} track",
    "laion": "This is a sound of {}.",
    "the_sound": "the sound of {}",
    "song": "{} song",
    "edm": "{} electronic dance music",
    "style": "music in the style of {}",
}

CAPTIONS = {
    "techno": "A driving four on the floor beat with a pounding kick drum, dark synth stabs and repetitive machine-like percussion.",
    "tech house": "A groovy four on the floor beat with a rolling bassline, crisp hi-hats and funky percussive loops.",
    "house": "A warm four on the floor groove with offbeat hi-hats, piano chords and a soulful bassline.",
    "deep house": "A smooth laid back groove with a warm round bassline, jazzy chords and soft pads.",
    "progressive house": "A slowly building groove with long evolving synth layers, a steady pulse and an emotional melodic lift.",
    "minimal": "A sparse stripped down beat with very few elements, small clicks, subtle percussion and lots of space.",
    "minimal techno": "A sparse repetitive machine beat with tiny clicks, subtle rhythmic changes and almost no melody.",
    "deep techno": "A dark hypnotic rolling beat with deep submerged bass and foggy reverberant textures.",
    "dub techno": "A hazy pulsing beat with echoing chords that decay into long dubby delay and warm tape hiss.",
    "acid": "A squelchy resonant bassline sweeping through a filter over a hard repetitive electronic beat.",
    "acid house": "A squelchy resonant bassline sweeping through a filter over a bouncy four on the floor groove.",
    "tribal": "Layers of hand drums and rolling polyrhythmic percussion over a steady pulse.",
    "tribal house": "Rolling hand drums and bongos, chanting and polyrhythmic percussion over a four on the floor beat.",
    "trance": "Soaring supersaw synth melodies, a fast rolling offbeat bass and long euphoric breakdowns.",
    "progressive trance": "Slowly evolving hypnotic arpeggios and a rolling offbeat bass with a long uplifting build.",
    "psy-trance": "Fast twisting acid synth lines and a rapid galloping bass with alien sound effects.",
    "breakbeat": "Syncopated broken drum loops with heavy sub bass and sampled drum breaks.",
    "electro": "Robotic drum machine rhythms, a staccato synth bassline and cold machine-like vocoder textures.",
    "industrial": "Harsh distorted metallic percussion, grinding noise and mechanical aggressive rhythms.",
    "ambient": "Slowly evolving atmospheric pads and textures with no beat and a calm spacious mood.",
    "downtempo": "A slow relaxed beat with mellow chords, soft bass and a dreamy atmosphere.",
    "disco": "A funky four on the floor beat with a bouncy electric bass, rhythm guitar, strings and brass stabs.",
    "nu-disco": "A funky retro groove with slap bass, analog synths and a bright rhythm guitar.",
    "funk": "A tight syncopated groove with slap bass, wah guitar and punchy brass hits.",
    "synth-pop": "Bright catchy analog synth melodies, a steady electronic drum beat and a pop vocal.",
    "ebm": "A stomping sequenced bassline, hard electronic drums and shouted vocals.",
    "uk garage": "A shuffling two-step beat with skippy hi-hats, pitched vocal chops and a warm sub bass.",
    "afrobeat": "Interlocking percussion, rhythmic guitars and bright horn lines over a hypnotic polyrhythmic groove.",
    "latin": "Syncopated congas and timbales, a bright brass section and a danceable latin rhythm.",
    "experimental": "Unusual abstract textures and noises with irregular structure and unconventional sounds.",
}

# Cada concepto dicho de varias maneras que para una persona significan lo mismo.
CONCEPTOS_PARAFRASIS = {
    "dark_hypnotic": ["dark hypnotic techno", "hypnotic dark techno", "techno that is dark and hypnotic", "deep, dark and trance-inducing techno",
                      "a dark hypnotic techno track", "gloomy and mesmerizing techno"],
    "happy_house": ["happy uplifting house", "uplifting happy house music", "joyful feel-good house", "cheerful house music that lifts your mood",
                    "a happy house track", "bright and euphoric house"],
    "industrial": ["heavy industrial techno", "industrial techno that is heavy", "hard mechanical industrial techno", "harsh and heavy factory-like techno",
                   "a heavy industrial techno track", "brutal industrial techno"],
    "vocal_soulful": ["soulful vocal house", "house music with soulful vocals", "gospel inspired vocal house", "house track with a soul singer",
                      "a soulful vocal house track", "warm house with a soulful female voice"],
    "melodic": ["emotional melodic techno", "melodic techno with emotion", "melancholic melodic techno", "sad and beautiful melodic techno",
                "a melodic techno track that is emotional", "deeply emotional techno with melodies"],
    "percussive": ["percussive groovy tech house", "tech house full of percussion", "tech house with lots of drums and groove", "rhythmic drum heavy tech house",
                   "a percussive tech house track", "groovy percussion-driven tech house"],
    "acid": ["acid bassline", "squelchy acid bass line", "a 303 acid line", "resonant acid synth bass", "acid house bassline", "TB-303 style squelching bass"],
    "chill": ["relaxed chill electronic music", "chilled out electronic music", "calm and relaxing electronic tracks", "laid back mellow electronic music",
              "a relaxing electronic track", "easygoing downtempo electronic music"],
    "aggressive": ["aggressive hard techno", "hard techno that is aggressive", "violent pounding hard techno", "fierce and intense hard techno",
                   "an aggressive hard techno track", "savage relentless techno"],
    "funky": ["funky disco groove", "disco with a funky groove", "groovy funk and disco", "a funky disco track", "bouncy funky retro disco", "dance floor funk and disco"],
    "wind": ["brass instruments", "horn section", "wind instruments", "trumpets and trombones", "brass band sound", "a brass section playing"],
    "piano": ["piano chords", "piano music", "a piano playing chords", "house piano riff", "piano melody", "uplifting piano stabs"],
}

NEGACION = ["techno", "house", "tech house", "melodic techno", "deep house", "dance music", "electronic music", "disco"]

IDIOMA = [
    ("techno oscuro", "dark techno"), ("música alegre", "happy music"), ("música triste", "sad music"), ("con piano", "with piano"),
    ("voz femenina", "female voice"), ("percusión tribal", "tribal percussion"), ("música de fiesta", "party music"), ("bajo profundo", "deep bass"),
    ("techno duro y agresivo", "hard aggressive techno"), ("instrumentos de viento", "wind instruments"), ("guitarra eléctrica", "electric guitar"),
    ("música relajante", "relaxing music"), ("house con voces", "house with vocals"), ("sintetizadores melódicos", "melodic synthesizers"),
    ("batería potente", "powerful drums"), ("ambiente oscuro", "dark atmosphere"),
]

# Una ocasión o idea y lo que esa ocasión suena, escrito a mano.
ACUSTICO = [
    ("birthday party music", "cheerful festive music with brass instruments and a bouncy rhythm"),
    ("birthday music with trumpets", "cheerful festive music with bright brass instruments"),
    ("wedding music", "romantic uplifting music with piano and strings"),
    ("sunset on the beach", "warm mellow music with soft guitar, relaxed percussion and gentle melodies"),
    ("driving at night", "steady hypnotic pulsing electronic music with dark synthesizers"),
    ("summer vibes", "bright sunny groovy music with warm chords and a light bouncy rhythm"),
    ("christmas", "music with sleigh bells, chimes and a warm festive choir"),
    ("after party at 6am", "slow deep hypnotic minimal music with sparse percussion"),
    ("gym workout", "high energy fast driving music with hard kicks and intense build-ups"),
    ("funeral", "slow solemn music with organ and dark sustained chords"),
    ("carnival", "fast joyful latin percussion, whistles and a bright brass section"),
    ("love", "warm tender music with a soft vocal and gentle chords"),
]

NULOS = ["tax invoice for the third quarter", "qwe zxc asd rty", "the cat sat on the table", "printer cartridge replacement",
         "photosynthesis in green plants", "a list of postal codes", "lorem ipsum dolor sit amet", "how to change a car tyre"]

LONGITUD = {
    "dark": ["dark techno",
             "dark hypnotic techno",
             "dark hypnotic techno with a deep rolling bassline",
             "dark hypnotic techno with a deep rolling bassline, sparse percussion and foggy reverberant textures",
             "dark hypnotic techno with a deep rolling bassline, sparse percussion, foggy reverberant textures, distant metallic stabs and a slow evolving tension that never fully releases"],
    "funky": ["funky house",
              "funky groovy house",
              "funky groovy house with a bouncy bassline",
              "funky groovy house with a bouncy bassline, disco strings and a chopped guitar riff",
              "funky groovy house with a bouncy bassline, disco strings, a chopped guitar riff, hand claps and a warm vintage sound that feels like a summer block party"],
    "melodic": ["melodic techno",
                "emotional melodic techno",
                "emotional melodic techno with soaring synths",
                "emotional melodic techno with soaring synths, warm pads and a slow melancholic arpeggio",
                "emotional melodic techno with soaring synths, warm pads, a slow melancholic arpeggio, wide reverb and a long cinematic breakdown that resolves into a huge drop"],
}

# Clase de AudioSet (AST) -> palabras candidatas, de la más específica a la más general.
INSTRUMENTOS = {
    "Trumpet": ["trumpet", "trumpets", "a trumpet playing", "corneta", "bugle", "horn"],
    "Brass instrument": ["brass instruments", "brass section", "horns", "horn section", "wind instruments", "brass"],
    "Saxophone": ["saxophone", "a saxophone playing", "sax solo", "wind instruments"],
    "Flute": ["flute", "a flute playing", "woodwind", "wind instruments"],
    "Wind instrument, woodwind instrument": ["wind instruments", "woodwind instruments", "flute", "saxophone"],
    "Piano": ["piano", "piano chords", "a piano playing", "keys"],
    "Electric piano": ["electric piano", "rhodes piano", "piano"],
    "Organ": ["organ", "church organ", "hammond organ"],
    "Synthesizer": ["synthesizer", "synth lead", "analog synth"],
    "Guitar": ["guitar", "a guitar playing", "guitar riff"],
    "Electric guitar": ["electric guitar", "distorted electric guitar", "guitar"],
    "Acoustic guitar": ["acoustic guitar", "spanish guitar", "guitar"],
    "Bass guitar": ["bass guitar", "slap bass", "bassline"],
    "Violin, fiddle": ["violin", "strings", "string section"],
    "Bowed string instrument": ["strings", "string section", "violins and cellos"],
    "Drum kit": ["drum kit", "live drums", "drums"],
    "Bass drum": ["kick drum", "bass drum", "pounding kick"],
    "Hi-hat": ["hi-hat", "hi hats", "open hi-hats"],
    "Snare drum": ["snare drum", "snare", "snare rolls"],
    "Tambourine": ["tambourine", "jingles"],
    "Cowbell": ["cowbell"],
    "Clapping": ["hand claps", "clapping", "claps"],
    "Bell": ["bells", "church bells", "chimes"],
    "Singing": ["singing", "a singer", "vocals"],
    "Female singing": ["female vocals", "a woman singing", "female singer", "diva vocals"],
    "Male singing": ["male vocals", "a man singing", "male singer"],
    "Choir": ["choir", "gospel choir", "choral voices"],
    "Rapping": ["rap", "rapping", "a rapper"],
    "Speech": ["spoken word", "a man talking", "speech"],
    "Whistling": ["whistling", "whistle"],
    "Crowd": ["crowd", "crowd cheering", "audience"],
    "Siren": ["siren", "police siren", "air horn"],
    "Vehicle horn, car horn, honking": ["air horn", "car horn", "horn"],
    "Laughter": ["laughter", "people laughing"],
    "Thunderstorm": ["thunder", "storm"],
    "Rain": ["rain", "rainfall"],
    "Ocean": ["ocean waves", "sea"],
    "Bird": ["birds singing", "birdsong"],
}

ARTISTAS = ["Charlotte de Witte", "Amelie Lens", "Adam Beyer", "Maceo Plex", "Solomun", "Tale Of Us", "Fisher", "Dubfire", "Len Faki", "Carl Cox",
            "Stephan Bodzin", "Boris Brejcha", "Marco Carola", "Hot Since 82", "Patrice Baumel", "Denis Sulta", "Nicolas Jaar", "Ben Böhmer"]

VOCABULARIO = {
    "genero": ["techno", "tech house", "house", "deep house", "minimal techno", "melodic techno", "acid house", "progressive house", "disco", "funk", "trance",
               "dub techno", "breakbeat", "electro", "afro house", "latin house", "ambient", "downtempo", "industrial techno", "hard techno"],
    "animo": ["dark", "happy", "sad", "melancholic", "euphoric", "uplifting", "hypnotic", "aggressive", "calm", "relaxed", "mysterious", "romantic", "energetic", "playful",
              "sensual", "epic", "dreamy", "tense", "joyful", "festive", "nostalgic", "cheerful"],
    "ritmo": ["four on the floor", "syncopated", "shuffle rhythm", "polyrhythmic", "driving beat", "broken beat", "swing groove", "rolling groove", "steady pulse", "slow tempo", "fast tempo",
              "bouncy groove", "half time"],
    "percusion": ["kick drum", "hand claps", "hi-hats", "cowbell", "congas", "bongos", "tribal drums", "shaker", "tambourine", "snare rolls", "percussion", "drum machine", "live drums", "808 bass drum", "909 drum machine"],
    "bajo": ["bassline", "sub bass", "deep bass", "acid bass", "slap bass", "distorted bass", "rolling bassline", "funky bass"],
    "armonia": ["piano chords", "organ", "strings", "synth pads", "arpeggio", "synth lead", "stabs", "electric piano", "melody", "harmonies"],
    "viento": ["brass instruments", "horns", "trumpet", "saxophone", "flute", "wind instruments", "woodwind", "trombone"],
    "cuerdas": ["electric guitar", "acoustic guitar", "guitar riff", "violin", "cello", "harp"],
    "voz": ["female vocals", "male vocals", "vocal chops", "spoken word", "choir", "gospel vocals", "rap vocals", "chanting", "whispering", "vocoder", "humming"],
    "textura": ["distorted", "reverb", "echo", "lo-fi", "warm analog", "metallic", "noisy", "clean", "airy", "gritty", "glitchy", "filtered", "atmospheric", "spacious", "raw"],
    "sonido_ambiente": ["crowd noise", "sirens", "bells", "birds", "rain", "ocean waves", "laughter", "whistles", "air horn", "gunshots", "explosions", "water"],
    "estructura": ["build-up", "breakdown", "big drop", "long intro", "peak time", "riser"],
    "ocasion": ["party", "birthday", "wedding", "club", "beach", "sunset", "festival", "carnival", "christmas", "after party", "gym", "road trip"],
}
