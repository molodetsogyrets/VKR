import pandas as pd
import spacy
from tqdm import tqdm
import re
from collections import defaultdict

nlp = spacy.load("ru_core_news_lg")

SCIENTIST_STEMS = [
    'учены', 'учено', 'исследовател', 'профессор', 'академик',
    'специалист', 'эксперт', 'биолог', 'физик', 'химик', 'астроном', 'математик', 'генетик',
    'нейробиолог', 'геолог', 'антрополог', 'палеонтолог', 'археолог', 'океанолог', 'климатолог',
    'микробиолог', 'вирусолог', 'иммунолог', 'нейрофизиолог', 'биофизик', 'биохимик', 'социолог', 'лингвист', 'ректор', 'декан',
    'кандидат', 'phd', 'постдок', 'аспирант', 'докторант', 'лаборатор'
]

SCIENTIST_PHRASES = [
    'научная группа', 'доктор наук', 'доктора наук','кандидат наук',
    'научный руководитель', 'научный сотрудник', 'ведущий ученый', 'коллектив ученых'
]

SCIENTIST_PATTERN = re.compile('|'.join(SCIENTIST_STEMS + SCIENTIST_PHRASES), re.IGNORECASE)

def contains_scientist_term(text):
    return bool(SCIENTIST_PATTERN.search(text))

def extract_scientist_contexts(text, window_size=100):
    contexts = []
    for match in SCIENTIST_PATTERN.finditer(text.lower()):
        start = max(0, match.start() - window_size)
        end = min(len(text), match.end() + window_size)
        contexts.append(text[start:end])
    return list(set(contexts))

def analyze_context(context):
    doc = nlp(context)
    analysis = {
        'active': 0,
        'passive': 0,
        'quotes': '«' in context or '"' in context,
        'verbs': defaultdict(int),
        'evidence': []
    }
    
    scientist_terms = []
    for match in SCIENTIST_PATTERN.finditer(context.lower()):
        term = context[match.start():match.end()]
        scientist_terms.append(term.lower())
    
    if not scientist_terms:
        return analysis
    
    for token in doc:
        if token.pos_ == 'VERB':
            subjects = []
            for child in token.children:
                if child.dep_ in ("nsubj", "agent"):
                    subjects.append((child, 'active'))
                elif child.dep_ in ("nsubj:pass", "obl", "dobj"):
                    subjects.append((child, 'passive'))
            
            for subject, construction_type in subjects:
                subject_text = subject.text.lower()
                
                is_scientist = False
                for term in scientist_terms:
                    if term in subject_text or subject_text in term:
                        is_scientist = True
                        break
                
                if not is_scientist:
                    is_scientist = bool(SCIENTIST_PATTERN.search(subject_text))
                
                if is_scientist:
                    if construction_type == 'active':
                        analysis['active'] += 1
                        analysis['verbs'][token.lemma_] += 1
                        analysis['evidence'].append(f"актив: {subject.text} {token.text} ({token.lemma_})")
                    else:
                        analysis['passive'] += 1
                        analysis['verbs'][token.lemma_] += 1
                        analysis['evidence'].append(f"пассив: {subject.text} {token.text} ({token.lemma_})")

    if analysis['active'] == 0 and analysis['passive'] == 0 and scientist_terms:
        analysis['active'] = 1
        analysis['evidence'].append(f"автоматически: найден термин '{scientist_terms[0]}'")
    
    return analysis

def analyze_all_news(file_path):
    """Анализирует все новости из файла"""
    df = pd.read_excel(file_path)
    print(f"Загружено {len(df)} строк")
    
    scientist_news_count = df['is_scientist_news'].sum()
    print(f"Новостей с is_scientist_news=1: {scientist_news_count}")
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=min(100, len(df)), desc="Анализ новостей"):
        if row['is_scientist_news'] != 1:
            continue
            
        text = row['text']
        if not isinstance(text, str) or not text.strip():
            continue
            
        contexts = extract_scientist_contexts(text)
        
        if not contexts:
            continue
            
        combined = {
            'active': 0,
            'passive': 0,
            'quotes': False,
            'verbs': defaultdict(int),
            'evidence': []
        }
        
        for context in contexts:
            analysis = analyze_context(context)
            combined['active'] += analysis['active']
            combined['passive'] += analysis['passive']
            combined['quotes'] = combined['quotes'] or analysis['quotes']
            for verb, count in analysis['verbs'].items():
                combined['verbs'][verb] += count
            combined['evidence'].extend(analysis['evidence'])
        
        if combined['active'] + combined['passive'] > 0:
            result_row = row.to_dict()
            ведущая_агентность = 'активная' if combined['active'] > combined['passive'] else \
                              'пассивная' if combined['passive'] > combined['active'] else 'смешанная'
            
            result_row.update({
                'найдены_термины': ', '.join(set(m.group() for m in SCIENTIST_PATTERN.finditer(text.lower()))),
                'активность': combined['active'],
                'пассивность': combined['passive'],
                'коэффициент_агентности': round(combined['active'] / (combined['active'] + combined['passive']), 2),
                'ведущая_агентность': ведущая_агентность,
                'цитирование': 'да' if combined['quotes'] else 'нет',
                'топ_глаголы': ', '.join(sorted(combined['verbs'], key=lambda x: combined['verbs'][x], reverse=True)[:3]),
                'примеры': ' | '.join(combined['evidence'][:3]) if combined['evidence'] else 'нет'
            })
            results.append(result_row)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    input_file = "C:/Users/Varvara.Chalenko/Desktop/1111/all.xlsx"
    output_file = "C:/Users/Varvara.Chalenko/Desktop/1111/анализ_всех_новостей.xlsx"
    
    print("Анализ всех новостей...")
    results_df = analyze_all_news(input_file)
    
    print(f"Размер результатов: {len(results_df)}")
    
    try:
        if not results_df.empty:
            print(f"Сохранение результатов в {output_file}...")
            with pd.ExcelWriter(output_file) as writer:
                results_df.to_excel(writer, sheet_name='Анализ', index=False)
                
                # Сводная статистика
                stats = pd.DataFrame({
                    'Всего новостей': [100],
                    'С учеными': [len(results_df)],
                    'С цитированием': [results_df['цитирование'].eq('да').sum()],
                    'Средний коэффициент агентности': [results_df['коэффициент_агентности'].mean()]
                })
                stats.to_excel(writer, sheet_name='Статистика', index=False)

    except Exception as e:
        print(f"Ошибка: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
