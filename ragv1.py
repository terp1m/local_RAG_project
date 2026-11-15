from ollama import chat
import ollama

dataset = []
with open('car.txt', 'r',encoding='utf-8') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')

EMBEDDING_MODEL = 'embeddinggemma-2'
LANGUAGE_MODEL = 'qwen3:8b'

VECTOR_DB = []
def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i,chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i + 1}/{len(dataset)} to the database')

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  similarities = []
  for chunk, embedding in VECTOR_DB:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  similarities.sort(key=lambda x: x[1], reverse=True)
  return similarities[:top_n]

#CHATBOT

while True:
  query = input('\nYou:')
  if query.strip().lower() == 'quit':
    break
  retrieved_knowledge = retrieve(query)
  instruction_prompt = f'''You are a helpful chatbot.
  Use only the following pieces of context to answer the question. Don't make up any new information:
  {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
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