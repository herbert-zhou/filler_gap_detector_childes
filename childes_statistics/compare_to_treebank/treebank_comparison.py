from treebank_inference_utils import *
from detection_algorithms.get_labels import get_labels
from tqdm import tqdm
import ast

def find_root_by_folder_name(folder_name: str, start: Path | None = None) -> Path:
    """
    Find the nearest ancestor directory whose name matches `folder_name`.

    Works when code is run from different subdirectories inside the project.
    """
    if start is None:
        # Use __file__ in scripts; fall back to cwd in notebooks/interpreters
        try:
            start = Path(__file__).resolve()
        except NameError:
            start = Path.cwd().resolve()
    else:
        start = Path(start).resolve()

    # If start is a file, begin from its parent
    if start.is_file():
        start = start.parent

    for path in [start, *start.parents]:
        if path.name == folder_name:
            return path

    raise FileNotFoundError(
        f"Could not find a parent directory named {folder_name!r} from {start}"
    )

ROOT_DIR = find_root_by_folder_name("filler_gap_detector_childes")
DATA_DIR = ROOT_DIR / "datasets"
CURR_DIR = ROOT_DIR / "childes_statistics" / "compare_to_treebank"


label_conversion_h2p = {
    # MATRIX_Q
    'MQ_SUBJ': ['SMQ', 'CC_SMQ'],
    'MQ_OBJ': ['OMQ', 'CC_OMQ'],
    'MQ_ADJUNCT': ['AMQ', 'CC_AMQ'],
    'MQ_POLAR': ['PMQ'],
    
    # EMBEDDED_Q
    'EQ_SUBJ': ['SEQ', 'SMQ', 'CC_SMQ', 'NA'],
    'EQ_OBJ': ['OEQ', 'OMQ', 'CC_OMQ', 'NA'],
    'EQ_ADJUNCT': ['AEQ', 'PMQ', 'NA', 'AMQ', "CC_AMQ"],
    'EQ_POLAR': ['PEQ', 'NA'],
    
    # REL_CLAUSE
    'RC_SUBJ': ['SRC', 'SRC_reduced', 'NA'],
    'RC_OBJ': ['ORC', 'ORC_reduced', 'NA'], 
    'RC_ADJUNCT': ['ARC', 'PRC', 'NA'], 
}

label_conversion_p2h = {
    # MATRIX_Q
    'SMQ': ['MQ_SUBJ'],
    'CC_SMQ': ['MQ_SUBJ', 'EQ_SUBJ','RC_SUBJ'],
    'OMQ': ['MQ_OBJ'],
    'CC_OMQ': ['MQ_OBJ', 'EQ_OBJ'],
    'AMQ': ['MQ_ADJUNCT'],
    'CC_AMQ': ['MQ_ADJUNCT', 'EQ_ADJUNCT'],
    'PMQ': ['MQ_POLAR'],
    # EMBEDDED_Q
    'SEQ': ['EQ_SUBJ'],
    'OEQ': ['EQ_OBJ'],
    'AEQ': ['EQ_ADJUNCT'],
    'PEQ': ['EQ_POLAR'],
    # RC
    'SRC': ['RC_SUBJ'], 
    'SRC_reduced': ['RC_SUBJ'],
    'ORC': ['RC_OBJ'],
    'ORC_reduced': ['RC_OBJ'],
    'ARC': ['RC_ADJUNCT', 'EQ_ADJUNCT'], 
    'PRC': ['RC_ADJUNCT'],
    }

######################################################################## 
# Helper functions for computing precision, recall, F1 
######################################################################## 
def find_precision(parser_label):
    valid_human_labels = label_conversion_p2h.get(parser_label, [])
    df_sub = df[ df['parser_labels'].apply(lambda labels: parser_label in labels)]
    n_TP = 0
    n_total = len(df_sub)
    for _, row in df_sub.iterrows():
        if any(h_label in valid_human_labels for h_label in row['human_labels']):
            n_TP += 1
    precision = n_TP / n_total if n_total > 0 else 0.0
    return precision, n_total

def find_recall(human_label):
    valid_parser_labels = label_conversion_h2p.get(human_label, [])
    df_sub = df[ df['human_labels'].apply(lambda labels: human_label in labels)]
    n_TP = 0
    n_total = len(df_sub)
    for _, row in df_sub.iterrows():
        if any(p_label in valid_parser_labels for p_label in row['parser_labels']):
            n_TP += 1
    recall = n_TP / n_total if n_total > 0 else 0.0
    return recall, n_total

def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

######################################################################## 
# Other helpers 
######################################################################## 
def to_list_of_str(x):
    # already list/tuple
    if isinstance(x, (list, tuple)):
        return [str(t).strip() for t in x if str(t).strip()]

    # missing
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # string that might encode a list
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.upper() == "NONE":
            return []
        # try parsing python-literal list like "['SMQ']"
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return [str(t).strip() for t in v if str(t).strip()]
        except Exception:
            pass
        # fallback: treat as a single label
        return [s]

    # any other type -> coerce to single string label
    return [str(x).strip()]

omitted_labels = []

def parse_human_labels(cell):
    if pd.isna(cell) or cell == "NONE":
        return []
    items = [x.strip() for x in str(cell).split(";") if x.strip()]
    return [x for x in items if (not x.startswith("OTHER")) and (x not in omitted_labels)]


if __name__ == "__main__":
    # 1. Process all treebank files
    for corpus in ['brown-adam4up', 'brown-adam3to4', 'brown-eve', 'valian']: #
        input_file = f'{DATA_DIR}/CHILDESTreebank-curr/{corpus}+animacy+theta.parsed'
        prefix = None
        process_file(Path(input_file), out_prefix=prefix)
        
    # 2. Merge the 4 targetcsv files into one big csv file, while adding a column 'corpus' in the dataframe
    df_merged = pd.DataFrame()
    for corpus in ['brown-adam4up', 'brown-adam3to4', 'brown-eve', 'valian']:
        df = pd.read_csv(f'{DATA_DIR}/CHILDESTreebank-curr/{corpus}+animacy+theta_tree_final_classification.csv')
        df['corpus'] = corpus
        df_merged = pd.concat([df_merged, df], ignore_index=True)
    df_merged.to_csv(f'{CURR_DIR}/sp13_all_data.csv', index=False)
    
    
    # 3. Get parser labels for all sentences in the merged csv file, and save to a new csv file with an additional column 'parser_labels'
    input_file = f'{CURR_DIR}/sp13_all_data.csv'
    df = pd.read_csv(input_file)
    utterances = df['sentence'].tolist()
    print(f"Total number of sentences: {len(utterances)}")

    parser_labels = []
    for _, sen in enumerate(tqdm(utterances, miniters=1000), 1):
        sen = sen.replace(" *NULL*", "")
        labels = get_labels(sen)
        parser_labels.append(labels)
    df["parser_labels"] = parser_labels
    
    output_file = f"{CURR_DIR}/sp13_all_data_PARSED_LABELS.csv"
    df.to_csv(output_file, index=False)
    
    df["human_labels"] = df["final_labels_str"].apply(parse_human_labels).astype(object)
    df["parser_labels"] = df["parser_labels"].apply(to_list_of_str).astype(object)
    df["human_labels"]  = df["human_labels"].apply(to_list_of_str).astype(object)   
    
    # 4. Compute precision, recall, F1 for each label and save to a new csv file
    for label in label_conversion_p2h:
        precision, n_total = find_precision(label)
        # if the precision greater than 0.8, print *** before the label to highlight it
        if precision > 0.8:
            print(f"*** Parser label: {label} | Precision: {precision:.3f} | Total: {n_total}")
        else:
            print(f"Parser label: {label} | Precision: {precision:.3f} | Total: {n_total}")
