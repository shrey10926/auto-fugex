import re, io
import spacy
import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_md")

nlp = load_spacy_model()


def base_regex_fn(disc):

    '''
    Removes special characters 
    adds escape characters for $
    tokenizes and removes stop words
    strips ".,", removes dupes and rejoins tokens
    '''

    clean_text = re.sub(r'[^a-zA-Z0-9,.$%]', ' ', disc.lower()).strip()
    regex_inter = re.sub(r'[$]', r'\$', clean_text)
    doc = nlp(regex_inter)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    seen = set()
    tok_list = [x.strip('.').strip(',') for x in filtered_tokens if not (x in seen or seen.add(x))]
    final = ' '.join((' '.join(tok_list)).split())
    inter_step = re.sub(r'\$\s+', '$', final)
    inter_step1 = re.sub(r'\s+%+', '%', inter_step)
    split = inter_step1.split()
    return split


def sentence_tokenize(disc):

    '''
    Same as base_regex_fn but tokenizes for sentences
    '''
    
    clean_text = re.sub(r'[^a-zA-Z0-9,.$%]', ' ', disc.lower()).strip()
    regex_inter = re.sub(r'[$]', r'\$', clean_text)
    doc = nlp(regex_inter)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    joined_filtered = ' '.join(filtered_tokens)
    filtered_doc = nlp(joined_filtered)
    sent_disc = [sent.text for sent in filtered_doc.sents]
    global_list = [i.split() for i in sent_disc]
    finall = []
    for i in global_list:
        seen = set()
        tok_list = [x.strip('.').strip(',') for x in i if not (x in seen or seen.add(x))]
        final = ' '.join((' '.join(tok_list)).split())
        inter_step = re.sub(r'\$\s+', '$', final)
        inter_step1 = re.sub(r'\s+%+', '%', inter_step)
        split = inter_step1.split()
        finall.append(split)
    return finall


def base_fuzzy_fn(disc):

    '''
    Removes special chars, tokenizes, removes stopwords and rejoins tokens
    '''
    
    clean_text = re.sub(r'[^a-zA-Z0-9,.$%]', ' ', disc.lower()).strip()
    doc = nlp(clean_text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    seen = set()
    tok_list = [x for x in filtered_tokens if not (x in seen or seen.add(x))]
    final = ' '.join((' '.join(tok_list)).split())
    inter_step = re.sub(r'\$\s+', '$', final)
    inter_step1 = re.sub(r'\s+%+', '%', inter_step)
    split = inter_step1.split()
    return split


def create_sublist(ip_list, win_size, overlap):

    '''
    takes list of tokens as ip and creates sub-disclosures. op as nested list.
    '''
    
    result = []
    if not isinstance(ip_list, list) or not all(isinstance(item, str) for item in ip_list):
        raise ValueError('input list must be a list of strings')
    if win_size <= 0:
        raise ValueError(f'Batch Size ({win_size}) must be greater than 0!')
    if overlap < 0:
        raise ValueError(f'Overlap ({overlap}) must be non negative!')
    if overlap >= win_size:
        raise ValueError(f'Overlap ({overlap}) must be less than Batch Size ({win_size})!')
    if len(ip_list) < win_size:
        raise ValueError(f'Disclosure length ({len(ip_list)}) must be greater than the Batch Size ({win_size})!')
    if not ip_list:
        return result
    i = 0
    while i < len(ip_list):
        sublist = ip_list[i : i + win_size]
        result.append(sublist)
        i += win_size - overlap
    if len(result[-1]) == 1:
        return result[:-1]
    else:
        return result


def fuzzy_process(ip_list):

    '''
    Creates fuzzy expressions dict from sub-disclosures
    '''
    
    final_dict = {}
    for idx, value in enumerate(ip_list):
        final_dict[f'fuzzy_expression{idx + 1}'] = ' '.join(value)
    return final_dict


def ordered_token_set(text):

    '''
    Fuzzy logic for token_set_ratio but with order
    '''
    
    words = text.split()
    seen = set()
    ordered_unique_words = [word for word in words if not (word in seen or seen.add(word))]
    return ' '.join(ordered_unique_words)


def ordered_token_set_fuzz_ratio(s1, s2):

    '''
    custom backend logic for ordered_token_set_ratio
    '''
    
    # Transform both strings using your ordered_token_set function
    s1_ordered = ordered_token_set(s1)
    s2_ordered = ordered_token_set(s2)
    # Now compare them using fuzz.ratio which is order-sensitive
    return fuzz.ratio(s1_ordered, s2_ordered)


def generate_expression(text_dict, algorithm):

    '''
    Generates fuzzy expressions from fuzzy expr dict
    '''
    
    final = {}
    for key, v in text_dict.items():
        words = v.split()
        if algorithm == "ratio":
            final[key] = v
        elif algorithm == "token_sort_ratio":
            final[key] = ' '.join(sorted(words))
        elif algorithm == "token_set_ratio":
            final[key] = ' '.join(sorted(set(words)))
        elif algorithm == "partial_ratio":
            final[key] = ' '.join(words[:min(5, len(words))])
        elif algorithm == "ordered_token_set":
            final[key] = ordered_token_set(v)
        else:
            raise ValueError("Invalid algorithm choice. Choose from: ratio, token_sort_ratio, token_set_ratio, partial_ratio, ordered_token_set.")
    return final


def regex_process(ip_list, pattern='(\\s*[A-Za-z0-9!@#$&?:()-.%+,\\/|]*\\s+){0,5}'):

    '''
    Generates regex expr.
    '''

    try:
        if ip_list:
            inter_dict, final_dict = {}, {}
            for idx, value in enumerate(ip_list):
                inter_dict[f'regex{idx + 1}'] = value
            for k, v in inter_dict.items():
                final_dict[k] = '(?i)' + '(' + pattern.join(v) + ')'
            return final_dict
    except Exception as e:
        print(f'{e}')
        return f'{e}'
    

def highlight_true(val):

    '''
    Function for highlighting sub disclosures
    '''
    
    return 'background-color: #C6EFCE; color: #006100' if val is True else ''


def validate_regex(regex_pattern):
    
    '''
    Attempt to compile the regex. Return (True, None) if valid; otherwise (False, error message).
    '''

    try:
        re.compile(regex_pattern)
        return True, None
    except re.error as e:
        return False, str(e)


def validate_fuzzy_expression(expression):

    """
    Check that the fuzzy expression is non-empty.
    """

    if not expression.strip():
        return False, "Fuzzy expression cannot be empty."
    return True, None


st.markdown("""
    <style>
    body { background-color: #f0f2f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .title { font-size: 2.5rem; font-weight: 700; color: #333333; text-align: center; }
    .description { font-size: 1.1rem; color: #555555; text-align: center; margin-bottom: 1.5rem; }
    .stTextArea > label { font-size: 1.2rem; font-weight: 600; }
    .stButton button { background-color: #007BFF; color: #ffffff; border: none; padding: 0.8rem 1.5rem; border-radius: 5px; font-size: 1rem; }
    .stButton button:hover { background-color: #0056b3; }
    mark { font-weight: bold; padding: 0.2em; border-radius: 3px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)





with st.container():
    st.markdown('<h1 class="title">Auto Chef</h1>', unsafe_allow_html=True)
    st.markdown('<p class="description">Enter your disclosure text below. Choose the tokenization method. For word level, set the batch size and overlap. Click <strong>Process</strong> to view the results.</p>', unsafe_allow_html=True)

    if st.button("❓ Need Help?"):
        st.markdown("""
        **How to Use This App**
        
        **1️⃣ Enter a disclosure**

        **2️⃣ Choosing a Tokenization technique**

        **3️⃣ Choosing a Fuzzy Matching Algorithm**

        **3️⃣ Test Generated Regex and Fuzzy Expressions**

        *For more help, contact support at email@email.com*
        """)

    disclosure = st.text_area("Disclosure Text", placeholder="Enter your disclosure text here...", value="")

    st.info("ℹ️ **Word Level Tokenization** removes duplicate words and ignores small words like 'the', 'is', etc. ℹ️**Sentence Level Tokenization** does the same but keeps full sentences intact.")
    st.warning("⚠️ **Sentence Level Tokenization** does not allow batch size and overlap selection.")

    tokenization_method = st.selectbox("Select Tokenization Method", 
                                         ["Word Level Tokenization", "Sentence Level Tokenization"])


    # Show batch size and overlap only for word level tokenization.
    if tokenization_method == "Word Level Tokenization":
        batch_size = st.number_input("Batch Size", min_value = 2, value = 5, step = 1, help = "Number of words per sub-disclosure. Cannot be 0!")
        overlap = st.number_input("Overlap", min_value = 0, max_value = 5, value = 1, step = 1, help = "Number of overlapping words. Cannot be negative and must be less than Batch Size!")
    else:
        # Clear any previous word-level parameters.
        batch_size = None
        overlap = None

    with st.expander("📌 What is a Fuzzy Matching Algorithm?"):
        st.write("""
        Fuzzy Matching helps find similar phrases, even if they are not identical.
        - **Ratio:** Direct similarity comparison.
        - **Token Sort Ratio:** Ignores word order.
        - **Token Set Ratio:** Removes duplicate words before comparison.
        - **Partial Ratio:** Matches shorter substrings.
        - **Ordered Token Set:** Same as Token Set Ratio but keeps the word order.
                         """)
    algo = st.selectbox("Select Fuzzy Logic Algorithm", 
                        ["ratio", "token_sort_ratio", "token_set_ratio", "partial_ratio", "ordered_token_set"])

    if st.button("Process"):
        if not disclosure:
            st.error("Please enter a disclosure.")
        else:
            try:
                # Save the tokenization method in session state.
                st.session_state['tokenization_method'] = tokenization_method
                # Also store overlap if word level.
                if tokenization_method == "Word Level Tokenization":
                    st.session_state['overlap'] = overlap
                    tokens = base_regex_fn(disclosure)
                    op1 = create_sublist(tokens, win_size=batch_size, overlap=overlap)
                else:  # Sentence Level Tokenization
                    # Clear any previous overlap value.
                    st.session_state.pop('overlap', None)
                    op1 = sentence_tokenize(disclosure)
            
                op2 = regex_process(op1)
                op3 = generate_expression(fuzzy_process(op1), algo)
            
                # Update session state with results.
                st.session_state['sub_disclosures'] = [" ".join(sublist) for sublist in op1]
                st.session_state['regex_dict'] = op2
                st.session_state['fuzzy_dict'] = op3
                st.session_state['selected_algo'] = algo

            except Exception as e:
                st.error(f"Error : {e}")
                print(f'aaaaa {e}')

# Display processed disclosures only if available
if 'sub_disclosures' in st.session_state:
    st.subheader("Sub Disclosures:")
    op1_strings = st.session_state['sub_disclosures']

    if st.session_state.get('tokenization_method') == "Word Level Tokenization":
        # Retrieve the overlap value from session state.
        word_level_overlap = st.session_state.get('overlap')
        colors = ["red", "yellow", "green", "blue", "orange", "purple", "cyan", "teal", "magenta", "lime"]
        op1_words = [s.split() for s in op1_strings]

        # Highlight overlapping words
        for i in range(len(op1_words) - 1):
            if len(op1_words[i]) >= word_level_overlap and len(op1_words[i+1]) >= word_level_overlap:
                color = colors[i % len(colors)]
                for j in range(word_level_overlap):
                    op1_words[i][-word_level_overlap + j] = f"<mark style='background-color: {color};'>{op1_words[i][-word_level_overlap + j]}</mark>"
                    op1_words[i+1][j] = f"<mark style='background-color: {color};'>{op1_words[i+1][j]}</mark>"

        op1_highlighted = [" ".join(words) for words in op1_words]
        for line in op1_highlighted:
            st.markdown(line, unsafe_allow_html=True)
    else:
        # For sentence-level tokenization, display without highlighting.
        for sub_disc in op1_strings:
            st.markdown(sub_disc)

# Editable Regex Expressions
if 'regex_dict' in st.session_state:
    st.subheader("Edit Generated Regex Patterns")
    edited_regex = {}
    for key, pattern in st.session_state['regex_dict'].items():
        edited_regex[key] = st.text_input(f"{key}", value=pattern)
    st.session_state['regex_dict'] = edited_regex

# Editable Fuzzy Expressions
if 'fuzzy_dict' in st.session_state:
    st.subheader("Edit Generated Fuzzy Patterns")
    edited_fuzzy = {}
    for key, expr in st.session_state['fuzzy_dict'].items():
        edited_fuzzy[key] = st.text_input(f"{key}", value=expr)
    st.session_state['fuzzy_dict'] = edited_fuzzy

# Testing Section for Regex and Fuzzy Expressions
st.markdown("### Test Generated Regex/Fuzzy on Your Data")
uploaded_file = st.file_uploader("Upload a CSV or Excel file (must contain a 'win_text' column)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f'Please upload an excel or csv file!')
        print(f'www --> {e}')

    
    if 'win_text' not in df.columns:
        st.error("The uploaded file must contain a 'win_text' column.")
    else:
        if 'regex_dict' not in st.session_state:
            st.error("Please process a disclosure first to generate regex expressions.")
        else:
            # Validate Regex Expressions
            validated_regex = {}
            regex_errors = False
            for key, pattern in st.session_state.get('regex_dict', {}).items():
                is_valid, error_message = validate_regex(pattern)
                if not is_valid:
                    st.error(f"Error in {key}: {error_message}. Please fix the pattern.")
                    regex_errors = True
                else:
                    validated_regex[key] = pattern
            if regex_errors:
                st.stop()  # Halt further processing if regex errors exist

            # Validate Fuzzy Expressions
            validated_fuzzy = {}
            fuzzy_errors = False
            for key, expr in st.session_state.get('fuzzy_dict', {}).items():
                is_valid, error_message = validate_fuzzy_expression(expr)
                if not is_valid:
                    st.error(f"Error in {key}: {error_message}")
                    fuzzy_errors = True
                else:
                    validated_fuzzy[key] = expr
            if fuzzy_errors:
                st.stop()  # Halt further processing if fuzzy expression errors exist

            df_regex = df.copy()

            try:
                for key, regex in validated_regex.items():
                    col_name = f"{key}_result"
                    df_regex[col_name] = df_regex['win_text'].astype(str).apply(lambda text: bool(re.search(regex, text)))
            except Exception as e:
                st.error(f'Error : {e}')
                print(f'SSSSS --> {e}')

            result_columns = [col for col in df_regex.columns if col.endswith('_result')]
            styled_df_regex = df_regex.style.applymap(highlight_true, subset=result_columns)
            output_regex = io.BytesIO()
            styled_df_regex.to_excel(output_regex, engine='openpyxl', index=False, sheet_name='Regex_Results')
            processed_data_regex = output_regex.getvalue()
            st.download_button(
                label="Download Regex Test Results Excel File", 
                data=processed_data_regex, 
                file_name="regex_test_results.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


        if 'fuzzy_dict' not in st.session_state:
            st.error("Please process a disclosure first to generate fuzzy expressions.")
        else:
            selected_algo = st.session_state.get('selected_algo', 'ratio')
            fuzzy_func_map = {
                "ratio": fuzz.ratio,
                "token_sort_ratio": fuzz.token_sort_ratio,
                "token_set_ratio": fuzz.token_set_ratio,
                "partial_ratio": fuzz.partial_ratio,
                "ordered_token_set": ordered_token_set_fuzz_ratio
            }

            try:
                fuzzy_func = fuzzy_func_map.get(selected_algo, fuzz.ratio)
            except Exception as e:
                st.error(f'Error : {e}')
                print(f'ffffizy --> {e}')

            df_fuzzy = df.copy()
            for key, fuzzy_expr in validated_fuzzy.items():
                col_name = f"{key}_score"
                df_fuzzy[col_name] = df_fuzzy['win_text'].astype(str).apply(
                    lambda text: fuzzy_func(text, fuzzy_expr)
                )
            output_fuzzy = io.BytesIO()
            df_fuzzy.to_excel(output_fuzzy, engine='openpyxl', index=False, sheet_name='Fuzzy_Results')
            processed_data_fuzzy = output_fuzzy.getvalue()
            st.download_button(
                label="Download Fuzzy Test Scores Excel File", 
                data=processed_data_fuzzy, 
                file_name="fuzzy_test_scores.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )