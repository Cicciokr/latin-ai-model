import re
from cltk.sentence.lat import LatinPunktSentenceTokenizer
from cltk.lemmatize.lat import LatinBackoffLemmatizer
from cltk.tokenizers import LatinTokenizationProcess
from cltk.alphabet.lat import normalize_lat
from cltk.data.fetch import FetchCorpus

corpus_downloader = FetchCorpus(language="lat")
#corpus_downloader.list_corpora = ['example_distributed_latin_corpus', 'lat_text_perseus', 'lat_treebank_perseus', 'lat_text_latin_library', 'phi5', 'phi7', 'latin_proper_names_cltk', 'lat_models_cltk', 'latin_pos_lemmata_cltk', 'latin_treebank_index_thomisticus', 'latin_lexica_perseus', 'latin_training_set_sentence_cltk', 'latin_word2vec_cltk', 'latin_text_antique_digiliblt', 'latin_text_corpus_grammaticorum_latinorum', 'latin_text_poeti_ditalia', 'lat_text_tesserae']
corpus_downloader.import_corpus("lat_models_cltk")
corpus_downloader.import_corpus('lat_text_latin_library')
corpus_downloader.import_corpus('latin_training_set_sentence_cltk')

def clean_corpus(
    input_file: str,
    output_file: str
):
    """
    Esegue i seguenti step:
    1. Rimozione righe che contengono testo pseudo-latino (es. 'lorem ipsum').
    2. Segmentazione in frasi con CLTK.
    3. Normalizzazione e lemmatizzazione di ogni frase.
    4. Filtraggio righe con regex specifica.
    5. Deduplicazione case-insensitive (più aggressiva).
    """

    # Inizializza tokenizer e lemmatizzatore della CLTK
    sentence_tokenizer = LatinPunktSentenceTokenizer()
    lemmatizer = LatinBackoffLemmatizer()

    # Utilizziamo un set per salvare le righe già viste in minuscolo
    seen_lines = set()

    # Regex per filtrare i caratteri ammessi
    valid_line_pattern = re.compile(r'^[A-Za-z0-9ÄÖÜäöüÆæŒœᵫĀāūōŌ\.\,\;\:\?\!\-\ Ęę]+$')

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for raw_line in f_in:
            line = raw_line.strip()

            # 1. Rimozione righe pseudo-latine (esempio con "lorem ipsum")
            if re.search(r'lorem\s+ipsum', line, re.IGNORECASE):
                continue

            # 2. Tokenizzazione in frasi (qui trattiamo la riga come singolo “blocco” di testo)
            sentences = sentence_tokenizer.tokenize(line)

            # 3. Normalizzazione e lemmatizzazione di ogni frase
            cleaned_sentences = []
            for sent in sentences:
                # Normalizza il testo
                normalized_sent = normalize_lat(sent)

                # Tokenizza la frase in parole
                tokens = normalized_sent.split()

                # Lemmatizza
                #lemmas = lemmatizer.lemmatize(tokens)
                #lemmas_only = [lemma for _, lemma in lemmas]

                #se non volessi applicare la lemmatizzazione
                cleaned_sentences.append(" ".join(tokens)) 
                #cleaned_sentences.append(" ".join(lemmas_only))

            # Ricostruisci la riga a partire dalle frasi lemmizzate
            cleaned_line = " ".join(cleaned_sentences).strip()

            # 4. Filtraggio con la regex
            if valid_line_pattern.match(cleaned_line):
                # 5. Deduplicazione case-insensitive (linee uguali a prescindere dal maiuscolo/minuscolo)
                lowercase_line = cleaned_line.lower()
                if lowercase_line not in seen_lines and cleaned_line != "":
                    f_out.write(cleaned_line + "\n")
                    seen_lines.add(lowercase_line)


if __name__ == "__main__":
    input_file_path = "./dataset/thelatinlibrary.txt"
    output_file_path = "./dataset/thelatinlibrary_cleaned.txt"

    clean_corpus(input_file_path, output_file_path)
    print(f"Pulizia completata. File pulito salvato in: {output_file_path}")