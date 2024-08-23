import gradio as gr
from peft import PeftModel
from dateutil import parser
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch
import torch.nn as nn
from fire import Fire

from transformers import AutoModel, AutoTokenizer


class myFAISS(FAISS):
    @classmethod
    def from_texts(
            cls,
            keys,
            values,
            embedding,
            metadatas=None,
            ids=None,
            **kwargs,
    ):
        embeddings = embedding.embed_documents(keys)
        return cls._FAISS__from(
            values,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )
    
    def add_texts(
        self,
        keys,
        values,
        metadatas=None,
        ids=None,
        **kwargs,
    ):
        embeddings = [self.embedding_function(k) for k in keys]
        return self._FAISS__add(values, embeddings, metadatas=metadatas, ids=ids)

class MyProcessor(nn.Module):

    def __init__(self, embedding_model_path, faiss_config):
        super(MyProcessor, self).__init__()
        self.embed = HuggingFaceEmbeddings(model_name=embedding_model_path, model_kwargs={'device': 'cpu'})
        self.faiss_config = faiss_config
        self.faiss_retriever = {}
        for db, db_cfg in self.faiss_config.items():
            try:
                vector_db = myFAISS.load_local(db_cfg['INDEX_PATH'], self.embed, db_cfg['INDEX_NAME'])
                print(f"[INFO] Load faiss db {db} successfully!")
            except Exception as e:
                print(f"[ERROR] {e}")
                vector_db = None
            self.faiss_retriever[db] = vector_db
    
    def add_qa_pairs(self, qa_dict, database="news"):
        queries = [k for k,_ in qa_dict.items()]
        answers = [v for _,v in qa_dict.items()]
        metadatas = [{
            "question": q
        } for q in queries]
        if not self.faiss_retriever[database]:
            self.faiss_retriever[database] = myFAISS.from_texts(queries, answers, self.embed, metadatas=metadatas)
            self.faiss_retriever[database].save_local(self.faiss_config[database]['INDEX_PATH'], self.faiss_config[database]['INDEX_NAME'])
        else:
            self.faiss_retriever[database].add_texts(queries, answers, metadatas=metadatas)

    def checkDate(self, query):
        # 检查 query 中是否包含日期，若没有，则补充当天日期
        try:
            parser.parse(query, fuzzy=True).date()
        except:
            import datetime
            now = datetime.datetime.now()
            current_hour = now.hour
            latest_date = (now.date() - datetime.timedelta(days=1)) if current_hour < 18 else now.date()
            query = query + f'（今天的日期是:{latest_date}）'
        return query
    
    def get_prompt(self, doc, query):
        prompt = """参考内容如下：
{docs}
作为个人知识答疑助手，请根据上述参考内容回答下面问题，你的数据都应该来自上述的参考内容，答案中不允许包含编造内容。
问题是：{query}，根据相关数据举例论证你的观点"""

        return prompt.format(docs=doc, query=query)
    
    def concat_docs(self, docs):
        if len(docs) <= 0:
            return "未检索到相关文档"
        result = ""
        for i, doc in enumerate(docs):
            result += f"[Doc {i+1}]\n{doc}\n\n"
        return result
    
    def process(self, query, database, topk, model, tokenizer, method):
        """ process the request data
        """
        query = self.checkDate(query)

        print(f"[INFO] User Query: {query}")
        docs = self.faiss_retriever[database].similarity_search_with_score(query, k=topk)
        docs = [doc[0].page_content for doc in docs]
        knowledge = self.concat_docs(docs)

        if method == '3':
            response = self.concat_docs(docs)
        else:
            user_prompt = self.get_prompt(knowledge, query)
            print(f"[INFO] Prompt: {user_prompt}")
            
            response, _ = model.chat(tokenizer, user_prompt, history=[])

        return response, knowledge


def getModel(base_model_path,
             lora_ckpt_path,
             embedding_model_path,
             faiss_config,
             device):
    # paramter worker_threads indicates concurrency of processing
    runner = MyProcessor(embedding_model_path,
                         faiss_config)

    model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_ckpt_path).to(device)
    model.eval()

    return model, tokenizer, runner

database_map = {
    '研报': 'reports',
    '市场数据': 'prices',
    '新闻': 'news'
}

method_map = {
    '仅通过一条知识进行分析': '1',
    '整合知识库进行分析': '2',
    "仅检索知识文档": '3'
}

def main(base_model_path="", lora_ckpt_path="", embedding_model_path=""):
    print(f"base_model_path: {base_model_path}")
    print(f"lora_ckpt_path: {lora_ckpt_path}")
    print(f"embedding_model_path: {embedding_model_path}")

    faiss_config = {
        "news": {
            "INDEX_PATH": "data/database_sample/news",
            "INDEX_NAME": "news_index"
        },
        "reports": {
            "INDEX_PATH": "data/database_sample/reports",
            "INDEX_NAME": "reports_index"
        },
        "prices": {
            "INDEX_PATH": "data/database_sample/prices",
            "INDEX_NAME": "prices_index"
        }
    }
    device = torch.device('cuda:0')
    model, tokenizer, runner = getModel(base_model_path,
                                        lora_ckpt_path,
                                        embedding_model_path,
                                        faiss_config,
                                        device)

    def respond(message, chat_history, topk_slider, database_radio, method_radio):
        database = database_map[database_radio]
        method = method_map[method_radio]
        output, knowledge = runner.process(
            query=message,
            database=database,
            topk=topk_slider,
            model=model,
            tokenizer=tokenizer,
            method=method
        )

        chat_history.append((message, output))
        return "", chat_history, knowledge

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                topk_slider = gr.Slider(
                    minimum=1, maximum=50, step=1, label="Top-K", value=5
                )
                database_radio = gr.Radio(
                    ['研报', '市场数据', '新闻'], label='所使用的数据库', value='研报'
                )
                method_radio = gr.Radio(
                    ['仅通过一条知识进行分析', '整合知识库进行分析', "仅检索知识文档"], label='知识库使用方法', value='仅通过一条知识进行分析'
                )
                knowledge_box = gr.Textbox(label='检索到的知识', value='暂无', interactive=False, lines=29)
            with gr.Column():
                chatbot = gr.Chatbot()
                chatbot.style(height=805)
        msg = gr.Textbox(label='请输入你的问题')
        clear = gr.Button("Clear")

        msg.submit(respond, [msg, chatbot, topk_slider, database_radio, method_radio], [msg, chatbot, knowledge_box])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    demo.launch()

if __name__ == '__main__':
    Fire(main)