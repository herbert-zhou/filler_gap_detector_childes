# for each word, use spacy to get its stem or lemma, save in a list
import spacy
import re
import json
import ast
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter

# lexical items involved in Maya's evaluation set

transitive_verbs = ["accept", "acknowledge", "admit", "aggravate", "answer", "arrest", "ask", "avoid", "bash", "beat", "bend", "bite", "bless", "bother", "break", "brush", "build", "bump", "burn", "call", "cancel", "capture", "carry", "catch", "change", "charge", "chase", "chastise", "check", "chill", "clean", "close", "clutch", "collect", "comfort", "confuse", "consume", "contradict", "convert", "copy", "correct", "cover", "crack", "cross", 'cut', "dampen", "dash", "daze", "dazzle", "deceive", "define", "delay", "deny", "derail", "describe", "destroy", "devastate", "dig", "discover", "discuss", "dismiss", "distinguish", 'disturb', 'drag', 'draw', 'dress', 'drink', 'drive', 'drop', 'drown', 'dry', 'dunk', 'eat', 'edify', 'eject', 'embarrass', 'embrace', 'empower', 'enable', 'enclose', 'encourage', 'enjoy', 'enlighten', 'enlist', 'entertain', 'escort', 'examine', 'excite', 'excuse', 'execute', 'fascinate', 'feed', 'feel', 'fight', 'file', 'fill', 'find', 'finish', 'fire', 'fix', 'flick', 'flip', 'follow', 'force', 'forget', 'forgive', 'freeze', 'frighten', 'fry', 'furnish', 'gather', 'get', 'grab', 'grasp', 'grease', 'grip', 'handle', 'hang', 'have', 'head', 'help', 'hide', 'highlight', 'hit', 'hoist', 'hold', 'honor', 'hug', 'hurry', 'hurt', 'imitate', 'impress', 'include', 'indulge', 'inform', 'insert', 'inspect', 'inspire', 'insure', 'interest', 'interrupt', 'interview', 'intimidate', 'involve', 'irritate', 'join', 'jolt', 'judge', 'keep', 'key', 'kick', 'kill', 'kiss', 'knock', 'lag', 'lay', 'lead', 'lean', 'leave', 'let', 'lick', 'lift', 'light', 'lighten', 'limit', 'link', 'list', 'load', 'lock', 'lose', 'love', 'lower', 'maintain', 'make', 'mark', 'marry', 'massage', 'melt', 'mix', 'mock', 'move', 'munch', 'name', 'notice', 'number', 'nurse', 'offend', 'open', 'order', 'own', 'pack', 'page', 'paralyze', 'park', 'pass', 'pay', 'persuade', 'petrify', 'pick', 'pierce', 'pin', 'place', 'play', 'please', 'poison', 'poke', 'possess', 'post', 'pour', 'prepare', 'press', 'print', 'promise', 'protect', 'pull', 'punch', 'punish', 'purchase', 'push', 'puzzle', 'question', 'quit', 'raid', 'raise', 'read', 'reassure', 'recognize', 'refill', 'relax', 'remind', 'remove', 'repel', 'replace', 'research', 'retard', 'retire', 'reveal', 'ride', 'ring', 'rip', 'rob', 'rub', 'run', 'satisfy', 'save', 'scan', 'scare', 'scold', 'scoop', 'scrub', 'seat', 'see', 'select', 'sell', 'send', 'set', 'sew', 'shake', 'shame', 'shift', 'shoot', 'shove', 'shut', 'sink', 'slam', 'slap', 'slice', 'slow', 'smell', 'smoke', 'snap', 'soak', 'soften', 'solve', 'sound', 'specify', 'speed', 'spell', 'spend', 'spill', 'spit', 'split', 'spoon', 'spread', 'squash', 'stab', 'stain', 'stake', 'start', 'startle', 'stay', 'steer', 'stir', 'stop', 'store', 'strike', 'study', 'stuff', 'suck', 'surprise', 'survey', 'swallow', 'switch', 'tape', 'taste', 'teach', 'tease', 'tell', 'tend', 'terrify', 'test', 'thank', 'threaten', 'throw', 'tickle', 'tie', 'tighten', 'tip', 'tire', 'toast', 'toss', 'touch', 'toe', 'transform', 'try', 'turn', 'tweak', 'twist', 'underestimate', 'understand', 'unload', 'unlock', 'untie', 'upgrade', 'use', 'vacate', 'videotape', 'vilify', 'violate', 'wake', 'want', 'warm', 'warn', 'wash', 'watch', 'wear', 'widen', 'win', 'wipe', 'wrack', 'wrap', 'wreck']

intransitive_verbs = ['quivered', 'faded', 'rested', 'proceeded', 'reflected', 'led', 'belonged', 'progressed', 'differed', 'yielded', 'glowed', 'interfered', 'began', 'whistled', 'voted', 'arose', 'fled', 'buzzed', 'volunteered', 'delighted', 'landed', 'yawned', 'acted', 'evolved', 'erupted', 'emerged', 'withdrew', 'squeaked', 'attended', 'partied', 'winked', 'rejoiced', 'leaped', 'shouted', 'flourished', 'roared', 'objected', 'intervened', 'insisted', 'cheered', 'lingered', 'stood', 'interrupted', 'trembled', 'apologized', 'recovered', 'escaped', 'fell', 'prevailed', 'faltered', 'sneezed', 'froze', 'listened', 'consented', 'jumped', 'appeared', 'exploded', 'relented', 'protested', 'complained', 'vanished', 'arrived', 'pounced', 'screamed', 'obeyed', 'yelped', 'gasped', 'tired', 'moved', 'persisted', 'paused', 'relaxed', 'hesitated', 'survived', 'giggled', 'collapsed', 'came', 'blinked', 'shivered', 'rose', 'grinned', 'cried', 'blushed', 'disappeared', 'quit', 'frowned', 'lied', 'groaned', 'sighed', 'stopped', 'succeeded', 'existed', 'laughed', 'smiled', 'nodded', 'agreed']#subcases for transitive verbs depending on whether they take animate, inanimate, neutral

animate = ['told', 'fed', 'paid', 'protected', 'trusted', 'challenged', 'adored', 'questioned', 'convinced', 'promised', 'encouraged', 'invited', 'mocked', 'guided', 'advised', 'complimented', 'rescued', 'instructed', 'reminded', 'informed', 'hired', 'greeted', 'appointed', 'entertained', 'rewarded', 'punished', 'blessed', 'summoned', 'assured', 'forgave', 'thanked', 'accompanied', 'escorted', 'persuaded', 'employed', 'notified', 'honored', 'scolded', 'interviewed', 'cautioned', 'befriended', 'reassured', 'congratulated', 'hugged', 'chastised', 'alerted']

inanimate = ['made', 'got', 'felt', 'wanted', 'cut', 'tried', 'built', 'wrote', 'enjoyed', 'bought', 'mentioned', 'fixed', 'experienced', 'explained', 'posted', 'designed', 'realized', 'threw', 'reported', 'opened', 'spread', 'ordered', 'shared', 'understood', 'denied', 'defined', 'pulled', 'sold', 'wore', 'caught', 'changed', 'filled', 'tied', 'handled', 'presented', 'prepared', 'rewrote', 'noticed', 'proposed', 'dropped', 'stated', 'tasted', 'published', 'purchased', 'locked', 'imagined', 'split', 'updated', 'dismissed', 'connected', 'admitted', 'printed', 'studied', 'rejected', 'passed', 'recorded', 'justified', 'remembered', 'loaded', 'boiled', 'sewed', 'grabbed', 'delivered', 'chopped', 'measured', 'packed', 'polished', 'cleaned', 'acknowledged', 'solved', 'edited', 'specified', 'debated']

inanimate_with_objects = [['made', 'a cake'], ['got', 'the gift'], ['felt', 'the cloth'], ['wanted', 'the gift'], ['cut', 'the cake'], ['tried', 'the dish'], ['built', 'the tower'], ['wrote', 'the book'], ['enjoyed', 'the show'], ['bought', 'the gift'], ['mentioned', 'the event'], ['fixed', 'the bike'], ['experienced', 'the show'], ['explained', 'the problem'], ['posted', 'the picture'], ['designed', 'the experiment'], ['realized', 'the dream'], ['threw', 'the ball'], ['reported', 'the story'], ['opened', 'the door'], ['spread', 'the news'], ['ordered', 'the package'], ['shared', 'the candy'], ['understood', 'the problem'], ['defined', 'the word'], ['denied', 'permission'], ['pulled', 'the rope'], ['sold', 'the book'], ['wore', 'the clothes'], ['caught', 'the ball'], ['changed', 'the color'], ['tied', 'the rope'], ['filled', 'the bucket'], ['handled', 'the problem'], ['presented', 'the talk'], ['prepared', 'dinner'], ['rewrote', 'the book'], ['noticed', 'the flower'], ['proposed', 'the idea'], ['dropped', 'the ball'], ['stated', 'the rule'], ['tasted', 'the food'], ['published', 'the book'], ['purchased', 'the book'], ['locked', 'the door'], ['imagined', 'the sunrise'], ['split', 'the apple'], ['updated', 'the post'], ['dismissed', 'the comment'], ['connected', 'the dots'], ['admitted', 'the mistake'], ['printed', 'the poster'], ['studied', 'the book'], ['rejected', 'the suggestion'], ['passed', 'the test'], ['recorded', 'the song'], ['justified', 'the action'], ['remembered', 'the song'], ['loaded', 'the car'], ['boiled', 'the water'], ['sewed', 'the shirt'], ['grabbed', 'the keys'], ['delivered', 'the package'], ['chopped', 'the vegetables'], ['measured', 'the height'], ['packed', 'the suitcase'], ['polished', 'the shoes'], ['cleaned', 'the closet'], ['acknowledged', 'the gift'], ['solved', 'the problem'], ['edited', 'the book'], ['specified', 'the rules'], ['debated', 'the issue']]

animate_present = ['call', 'devastate', 'forgive', 'reassure', 'startle', 'confuse', 'mock', 'rob', 'interview', 'disturb', 'judge', 'punish', 'imitate', 'empower', 'impress', 'fire', 'protect', 'chastise', 'offend', 'thank', 'marry', 'enlighten', 'comfort', 'intimidate', 'arreste', 'bless', 'scare', 'inform', 'contradict', 'enlist', 'destroy', 'kiss', 'hug', 'vilify', 'repel', 'join', 'persuade', 'aggravate', 'deceive', 'warn', 'remind', 'help', 'escort', 'question', 'tell', 'irritate', 'embarrass', 'bother', 'scold', 'surprise', 'interrupt', 'frighten', 'threaten', 'honor']

inanimate_present = ['lighten', 'print', 'spill', 'change', 'spend', 'cancel', 'understand', 'dry', 'order', 'pour', 'prepare', 'wear', 'want', 'sell', 'spread', 'try', 'pack', 'specify', 'build', 'copy', 'fix', 'taste', 'tighten', 'smell', 'get', 'split', 'notice', 'drink', 'admit', 'solve', 'widen', 'make', 'store', 'purchase', 'feel', 'enclose', 'pass', 'refill', 'post', 'grasp', 'pull', 'sew', 'maintain', 'study', 'enjoy']

verbs_with_that = ["accepted", "announced", "checked", "considered", "decided", "discovered", "forgot", "guessed", "imagined", "knew", "noticed", "proved", "remembered", "said", "saw", "understood"]

verbs_with_what = ["announced", "discovered", "forgot", "guessed", "knew", "remembered", "saw"]

infinitival_verbs = ["decided", "forgot", "remembered"]

nouns = ["you", "I", "we", "they", "he", "she", "it", "the doctor", "the scientist", "the person", "the singer", "the teacher", "the student", "the parent", "the child", "the writer", "the artist", "the friend", "the sibling", "John", "Mary", "Alex", "the neighbor", "Louis", "Catherine", "the astronaut"]

objects = ["you", "me", "us", "them", "him", "her", "it", "the doctor", "the scientist", "the person", "the singer", "the teacher", "the student", "the parent", "the child", "the writer", "the artist", "the friend", "the sibling", "John", "Mary", "Alex", "the neighbor", "Louis", "Catherine", "the astronaut"]

def remove_brackets(sentence: str) -> str:
    # Remove [ ... ] including the brackets
    cleaned = re.sub(r"\[.*?\]", "", sentence)
    # Collapse extra whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Fix stray spaces before punctuation
    cleaned = re.sub(r"\s+([?.!,])", r"\1", cleaned)
    return cleaned

def load_dataset(dataset, child_filtered, construction):
    '''
    Load the dataset and filter based on speaker and construction type.
    `construction` is either a string (level 0 or 2) or a list of strings (level 1).
    '''
    file_path = f'all_labeled_data/data_Nov15/LABELED_{dataset}.csv'
    df = pd.read_csv(file_path)
    df['labels'] = df['labels'].apply(ast.literal_eval)
    
    if child_filtered:
        df = df[df['speaker'] == 'PAR']
    else:
        df = df[df['speaker'] == 'CHI']
        
    if construction == 'All_Sentences': # "all" means get all sentences, no filtering
        utterances = df['sentence_clean'].tolist()
    elif construction == 'None_Construction': # "none" means sentences without any target construction
        # utterances = df[df['labels'] == []]['sentence_clean'].tolist()
        utterances = df[df['labels'].apply(lambda x: len(x) == 0)]['sentence_clean'].tolist()
    else:
        if isinstance(construction, str):
            construction = [construction]
        utterances = []
        for _, row in df.iterrows():
            labels = row['labels']
            if any(label in construction for label in labels):
                utterances.append(row['sentence_clean'])
    return utterances
    
def count_all_word_frequency(utterances):
    counts = Counter()
    # text_clean = [remove_brackets(utt) for utt in utterances]
    
    all_text = " ".join(utterances)
    all_text = re.sub(r"[^\w\s]", " ", all_text)  # remove punctuation
    all_words = all_text.split()  # split into words
    counts.update(Counter(all_words))  # get the raw frequency of each unique word in the utterances

    return counts

def extract_lexical_frequency(counts, unique_words, json_path, condition_label):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing JSON if it exists and is non-empty; otherwise start fresh
    if not json_path.exists() or json_path.stat().st_size == 0:
        data = {}
    else:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    # Update entries
    for word in unique_words:
        frequency = counts.get(word, 0)
        if word not in data or not isinstance(data[word], dict):
            data[word] = {}
        data[word][condition_label] = frequency

    # Save updated JSON
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

Verb_Categories = [transitive_verbs, 
                   intransitive_verbs, 
                   animate, 
                   inanimate, 
                   animate_present, 
                   inanimate_present, 
                   verbs_with_that, 
                   verbs_with_what, 
                   infinitival_verbs]

Noun_Categories = [[item.split()[-1] for item in nouns], 
                   [item.split()[-1] for item in objects],
                   [item[1].split()[-1] for item in inanimate_with_objects]]

Construction_Categories = {'MQ': ['SMQ', 'OMQ', 'CC_SMQ', 'CC_OMQ'],
                           'EQ': ['SEQ', 'OEQ'],
                           'RC': ['SRC', 'ORC', 'SRC_reduced', 'ORC_reduced']}
Construction_Flat = ['SMQ', 'OMQ', 'CC_SMQ', 'CC_OMQ', 'SEQ', 'OEQ', 'SRC', 'ORC', 'SRC_reduced', 'ORC_reduced',
                     'MQ', 'EQ', 'RC',
                     'None_Construction', 'All_Sentences']


# get the set of unique words from the lists above
unique_verbs = set()
for word_list in Verb_Categories:
    for word in word_list:
        unique_verbs.add(word)
# convert the set back to a list
unique_verbs = list(unique_verbs)
len(unique_verbs)

for dataset in ['dev', 'test', 'train_100M']: #'train_10M', 
    for child_filtered in [True, False]:
        file_child = 'Child' if not child_filtered else 'noChild'
        print(f"#################### On Dataset {dataset}, Filtered {child_filtered} #################### ")
        
        for construction in Construction_Flat:
            if construction in ['MQ', 'EQ', 'RC']: target_construction = Construction_Categories[construction]
            else: target_construction = construction
            
            json_path = f'dataset_lexical_frequency/EvalTokens_{dataset}_{file_child}.json'
            utterances = load_dataset(dataset, child_filtered, target_construction)
            all_counts = count_all_word_frequency(utterances)
            extract_lexical_frequency(all_counts, unique_verbs, json_path, construction)
            print(f"finished construction {construction}")
            
            if construction == 'All_Sentences':
                with open(f'dataset_lexical_frequency/AllTokens_{dataset}_{file_child}.json', 'w') as f:
                    json.dump(all_counts, f)
            