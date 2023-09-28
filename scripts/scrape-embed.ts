import { Document } from 'langchain/document';
import { NotionLoader } from 'langchain/document_loaders';
import type { SupabaseClient } from '@supabase/supabase-js';
import { Embeddings, OpenAIEmbeddings } from 'langchain/embeddings';
import { SupabaseVectorStore } from 'langchain/vectorstores';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { supabaseClient } from '../libs/supabase-client';

const directoryPath = "./config";

async function extractData(): Promise<Document[]> { 
  try {
    const loader = new NotionLoader(directoryPath);
    const docs = await loader.load();
    return docs;
  } catch (error) {
    console.error(`Error while extracting data: ${error}`);
    return [];
  }
}

async function embedDocuments(
  client: SupabaseClient,
  docs: Document[],
  embeddings: Embeddings,
) {
  console.log('creating embeddings...');
  await SupabaseVectorStore.fromDocuments(client, docs, embeddings);
  console.log('embeddings successfully stored in supabase');
}

async function splitDocsIntoChunks(docs: Document[]): Promise<Document[]> {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 200,
  });
  return await textSplitter.splitDocuments(docs);
}

(async function run() {
  try {
    //load data from each url
    const rawDocs = await extractData();
    //split docs into chunks for openai context window
    const docs = await splitDocsIntoChunks(rawDocs);
    //embed docs into supabase
    await embedDocuments(supabaseClient, docs, new OpenAIEmbeddings());
  } catch (error) {
    console.log('error occured:', error);
  }
})();
