import pandas as pd
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    PER, LOC, ORG,
    Doc
)
from razdel import tokenize
from tqdm import tqdm
from collections import defaultdict
import pymorphy3
import re

def clean_company_name(name):
    patterns = [
        r'^(ПАО|АО|ООО|ЗАО|НКО|АКБ|ИП|ОАО|ОАО|АО|ПАО|НАО|МКК|ГУП|ФГУП)\s+',
        r'\s+"([^"]+)"',
        r'\([^)]*\)'
    ]
    for pattern in patterns:
        name = re.sub(pattern, '', name).strip()
    return name

def normalize_person_name(name, morph):
    parts = name.split()
    if len(parts) >= 1:
        parsed = morph.parse(parts[-1])[0]
        if 'Surn' in parsed.tag:
            return parsed.normal_form.title()
    return name.title()

def extract_entities(text):
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        segmenter = Segmenter()
        morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        morph_tagger = NewsMorphTagger(emb)
        ner_tagger = NewsNERTagger(emb)
        morph_analyzer = pymorphy3.MorphAnalyzer()
        
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.tag_ner(ner_tagger)
        
        raw_entities = []
        for span in doc.spans:
            try:
                raw_entities.append({
                    'text': span.text,
                    'type': {PER: "PER", LOC: "LOC", ORG: "ORG"}.get(span.type, "OTHER")
                })
            except Exception as e:
                continue
        
        normalized_groups = defaultdict(list)
        
        for entity in raw_entities:
            try:
                if entity['type'] == 'PER':
                    normalized = normalize_person_name(entity['text'], morph_analyzer)
                elif entity['type'] == 'ORG':
                    normalized = clean_company_name(entity['text'])
                else:
                    parsed = morph_analyzer.parse(entity['text'])[0]
                    if any(tag in parsed.tag for tag in {'Geox', 'Name', 'Surn'}):
                        normalized = parsed.normal_form.title()
                    else:
                        normalized = entity['text']
                
                normalized = re.sub(r'[«»"\'()]', '', normalized).strip()
                normalized_groups[(normalized, entity['type'])].append(entity['text'])
                
            except Exception as e:
                continue
        
        entities = []
        for (norm_name, ent_type), originals in normalized_groups.items():
            entities.append({
                'original': originals[0],
                'normalized': norm_name,
                'type': ent_type,
                'surname': norm_name.split()[-1] if ent_type == 'PER' else '',
                'count': len(originals)
            })
        
        return entities
    
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        return []

def format_entities(entities):
    if not entities:
        return "", 0, "", "", "", ""
    
    entity_groups = defaultdict(list)
    surnames = set()
    
    for ent in entities:
        key = (ent['normalized'], ent['type'])
        entity_groups[key].append(ent)
        if ent['type'] == 'PER' and ent['surname']:
            surnames.add(ent['surname'])
    
    grouped_entities = []
    for (name, type_), items in entity_groups.items():
        total_count = sum(item['count'] for item in items)
        grouped_entities.append({
            'name': name,
            'type': type_,
            'count': total_count
        })
    
    sorted_entities = sorted(grouped_entities, 
                           key=lambda x: (-x['count'], x['name']))
    
    unique_entities = [e['name'] for e in sorted_entities]
    entities_with_types = [f"{e['name']} ({e['type']})" for e in sorted_entities]
    unique_with_count = [f"{e['name']} ({e['count']})" for e in sorted_entities]
    
    return (
        ", ".join(unique_entities),
        len(unique_entities),
        ", ".join(entities_with_types),
        ", ".join(unique_with_count),
        ", ".join(sorted(surnames)),
        len(surnames)
    )

file_path = r"C:\Users\Varvara.Chalenko\Desktop\1111\all.xlsx"
df = pd.read_excel(file_path)

if len(df) > limit:
    df = df.head(limit)
    print(f"Ограничено до {limit} новостей")
else:
    print(f"Всего новостей для обработки: {len(df)}")


tqdm.pandas(desc="Обработка новостей")
df['entities'] = df['text'].progress_apply(extract_entities)


df[['unique_entities', 'entities_count', 'entities_with_types', 
    'unique_entities_with_count', 'unique_surnames', 'surnames_count']] = df['entities'].apply(
    lambda x: pd.Series(format_entities(x))
)

output_path = file_path.replace('.xlsx', '_processed_new.xlsx')
df.to_excel(output_path, index=False)
