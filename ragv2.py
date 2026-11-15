import ollama
from ollama import chat
dataset = []

with open('car.txt','r',encoding='utf-8') as f:
    dataset = f.read()
def chunk_text(text,chunk_size = 200,overlap = 50):
    chunks = []
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap>=chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    difference_size = chunk_size - overlap
    start = 0
    id = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append({
            'chunk_id': id,
            'content':text[start:end],
            'start':start,
            'end':end
        })
        start += difference_size
        id += 1
    return chunks
EMBEDDING_MODEL = 'embeddinggemma-2'
LANGUAGE_MODEL = 'qwen3:8b'

VECTOR_DB = []
def get_embedding(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  return embedding
chunks = chunk_text(dataset)
for i,chunk in enumerate(chunks):
    embedding =get_embedding(chunk['content'])
    VECTOR_DB.append({
        'id': chunk['chunk_id'],
        'content': chunk['content'],
        'embedding': embedding,
        'start': chunk['start'],
        'end': chunk['end']

    })
    print(f'Added chunk {i + 1}/{len(chunks)} to the database')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  denominator = norm_a * norm_b
  if denominator == 0:
      return 0
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=9):
  query_embedding = get_embedding(query)
  similarities = []
  for a in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, a['embedding'])
    similarities.append((a, similarity))
  similarities.sort(key=lambda x: x[1], reverse=True)
  return similarities[:top_n]

while True:
  query = input('\nYou:')
  if query.strip().lower() == 'quit':
    break
  retrieved_knowledge = retrieve(query)

  context = '\n'.join(
      [f"- {chunk['content']}" for chunk, similarity in retrieved_knowledge]
  )

  instruction_prompt = f'''You are a helpful chatbot.
  Use only the following pieces of context to answer the question.
  Don't make up any new information:

  {context}
  '''
  stream = chat(
    model=LANGUAGE_MODEL,
    messages = [
    {'role':'system','content': instruction_prompt},
    {'role': 'user', 'content': query},
    ],
    stream=True
  )
  print("\nAnswer is ready:\n")
  for a in stream:
    print(a['message']['content'],end='',flush=True)