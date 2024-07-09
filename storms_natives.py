import json
import os
from uuid import uuid4
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser

from AI_docs.ai_docs_cfg import ZHIPU_AK
from volcenginesdkarkruntime import Ark

from AI_docs.ai_docs_cfg import ARK_AKEY, ARK_SKEY, DOUBAO_AGENT

# Langsmith 配置
# unique_id = uuid4().hex[0:8]
# os.environ["LANGCHAIN_PROJECT"] = f" Storm 文章创作 topic[{example_topic}] - {unique_id}"
# os.environ["LANGCHAIN_TRACING_V2"] = 'true'
# os.environ["LANGCHAIN_API_KEY"] = os.getenv('MY_LANGCHAIN_API_KEY')

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate

# Storm 配置

# 文章标题
example_topic = "How a Finacial Therapist Can Ease your money worries"

# ----- 嵌入模型配置
# 本地嵌入模型路径
LOCAL_EMBEDDING_MODEL_PATH = "D:\LLM\\bce_modesl\\bce-embedding-base_v1"

# 默认使用GPU
EMBEDDING_MODEL_KWARGS = {'device': 'cuda:0'}
# 使用 CPU
#EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
EMBEDDING_ENCODE_KWARGS = {'batch_size': 32, 'normalize_embeddings': True, }

# ----- Chroma 向量数据库配置
CHROMA_DB_PATH = "D:\\LLM\\my_projects\\chroma_db\\ai_docs"

# ----- Glm4
# Zhipu AI API Key
# 设置你的 ZHIPU_AK ： ZHIPU_AK='xxxxxxxxxxxxxxxx'


# ----- 豆包搜索引擎
# 你的火山引擎 ARK_AKEY 和 ARK_SKEY
# ARK_AKEY='xxxxxxxxxxxxxxxx'
# ARK_SKEY='xxxxxxxxxxxxxxxx'


from langchain_core.runnables import RunnableLambda

if __name__ == '__main__':

    # 嵌入模型配置
    embedding_model_name = LOCAL_EMBEDDING_MODEL_PATH
    embedding_model_kwargs = EMBEDDING_MODEL_KWARGS
    embedding_encode_kwargs = EMBEDDING_ENCODE_KWARGS

    # 嵌入模型初始化
    embed_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs
    )

    # Chroma 数据库配置
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embed_model)

    # 检索器
    retriever = vector_store.as_retriever()

    import threading
    rlock = threading.RLock()

    question_num = 0

    # 智谱清言
    glm4_air_model = ChatOpenAI(
        model_name="gLm-4-air",
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=ZHIPU_AK,

    )



    # 初步大纲生成 chain 模型
    init_ouline_model = glm4_air_model

    # 扩展topic chain 模型
    expand_topic_model = glm4_air_model

    # 角色生成 chain 模型
    gen_perspective_model = glm4_air_model

    # 特定角色提问 chain 模型
    gen_question_model = glm4_air_model

    # 根据对话，生成用于搜索引擎的问题 chain 模型
    create_question_model = glm4_air_model

    # 专家回答问题chain  模型
    ans_questions_model = glm4_air_model

    # 精炼大纲chain 模型
    refine_outline_model = glm4_air_model

    # 文字书写 chain 模型
    section_writer_model = glm4_air_model

    # 总结问题chain模型
    sum_queries_model = glm4_air_model

    # 豆包搜索引擎
    @tool
    def doubao_search(quert: str):
        """
        豆包搜索引擎，用于互联网搜索
        """

        client = Ark(ak=ARK_AKEY, sk=ARK_SKEY)

        print("----- 豆包搜索引擎-----")
        print('问题：{}'.format(quert))
        try:
            completion = client.bot_chat.completions.create(
                model=DOUBAO_AGENT,
                messages=[
                    {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能搜索引擎"},
                    {"role": "user", "content": quert},
                ],
            )
            res = completion.choices[0].message.content
        except Exception as e:
            print('豆包搜索出现异常 : {}'.format(e))
            res = '没有搜到相关内容'

        print('回答：{}'.format(res))

        return res
    # [测试点1] ： 验证你的豆包搜索引擎是否正常
    # check = doubao_search.batch(['五一长假','春节'])
    # pass

    # 初步大纲生成 提示词
    direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a multilingual writer proficient in multiple languages. Please create an outline for a topic provided by the user. It should be comprehensive and specific."
            ),
            ("user", "{topic}"),
        ]
    )

    # 生成初步提纲
    class Subsection(BaseModel):
        subsection_title: str = Field(..., title="Title of the subsection")
        description: str = Field(..., title="Content of the subsection")

        @property
        def as_str(self) -> str:
            return f"### {self.subsection_title}\n\n{self.description}".strip()

    class Section(BaseModel):
        section_title: str = Field(..., title="Title of the section")
        description: str = Field(..., title="Content of the section")
        subsections: Optional[List[Subsection]] = Field(
            default=None,
            title="Titles and descriptions for each subsection of the Wikipedia page.",
        )

        @property
        def as_str(self) -> str:
            subsections = "\n\n".join(
                f"### {subsection.subsection_title}\n\n{subsection.description}"
                for subsection in self.subsections or []
            )
            return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()

    class Outline(BaseModel):
        page_title: str = Field(..., title="Title of the Wikipedia page")
        sections: List[Section] = Field(
            default_factory=list,
            title="Titles and descriptions for each section of the Wikipedia page.",
        )

        @property
        def as_str(self) -> str:
            sections = "\n\n".join(section.as_str for section in self.sections)
            return f"# {self.page_title}\n\n{sections}".strip()

    # [测试点2] : 生成最初的大纲 chain
    direct_gen_outling_chain = direct_gen_outline_prompt | init_ouline_model
    initial_outline = direct_gen_outling_chain.invoke({"topic": example_topic})
    print('生成初步提纲: {}'.format(initial_outline))
    pass

    # 扩展主题 提示词
    gen_related_topics_prompt = ChatPromptTemplate.from_template(
        """
        I am writing an article on a topic mentioned below. 
        Please identify and recommend some content closely related to the topic. 
        I am looking for examples that provide insights into interesting aspects commonly associated with this topic, 
        or examples that help me understand the typical content and structure included in pages for similar topics. 
        Please list as many subjects as you can. Return subjects list only.
        Topic of interest:
        {topic}
        """
    )

    class RelatedSubjects(BaseModel):
        topics: List[str] = Field(
            description="Comprehensive list of related subjects as background research.",
        )

    # 根据主题，扩展内容 chain
    expand_chain = gen_related_topics_prompt | expand_topic_model

    # 角色定义
    class Editor(BaseModel):
        affiliation: str = Field(
            description="Primary affiliation of the editor.",
        )
        name: str = Field(
            description="Name of the editor.",pattern=r"^[a-zA-Z0-9_-]{1,64}$"
        )
        role: str = Field(
            description="Role of the editor in the context of the topic.",
        )
        description: str = Field(
            description="Description of the editor's focus, concerns, and motives.",
        )

        @property
        def persona(self) -> str:
            return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


    # 角色列表
    class Perspectives(BaseModel):
        editors: List[Editor] = Field(
            description="Comprehensive list of editors with their roles and affiliations.",
            # Add a pydantic validation/restriction to be at most M editors
        )

    # 生成多角色 提示词
    gen_perspectives_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You need to select a diverse (and distinct) group of editorial team members who will collaborate to create a comprehensive article on the topic. 
                Each of them represents a different perspective, role, or affiliation related to this topic. 
                Return below format:
                ------
                Editor:
                affiliation:
                description:
                name:
                role:
                ------
                You can draw inspiration from other articles on related topics. For each editorial team member, please add a description of what they will focus on.
                Outline of the content pages on related topics for inspiration:
                {examples}
                ------
                Generate formatted data pattern as below:
                {data_pattern}  
                """,
            ),
            ("user", "Topic of interest: {topic}"),
        ]
    )

    # 结构化数据提取
    perspectives_parser = PydanticOutputParser(pydantic_object=Perspectives)
    perspectives_data_pattern = perspectives_parser.get_format_instructions()
    gen_perspectives_chain = gen_perspectives_prompt | gen_perspective_model | perspectives_parser

    # 根据topic，生成多个扩展内容
    def survey_subjects(topic: str):
        related_subjects = expand_chain.invoke({"topic": topic})
        related_subjects_list = related_subjects.content.split('\n')
        subjects_num = len(related_subjects_list)
        print('生成了 {} 个扩展topic'.format(subjects_num))

        # 利用搜索引擎，搜索相关内容
        # 注意这里默认只进行有限搜索 ：related_subjects_list[:1]
        # 全部搜索：related_subjects_list[:1] -> related_subjects_list
        raw_search = doubao_search.batch(related_subjects_list[:1],{"max_concurrency": 3})

        # 搜索内容合并
        raw_text = ''
        for item in raw_search:
            raw_text+=item+'\n'

        # 生成角色
        res = gen_perspectives_chain.invoke({"examples": raw_text, "topic": topic, "data_pattern": perspectives_data_pattern})

        editor_num = len(res.editors)
        print('生成了 {} 个 角色'.format(editor_num))
        for i in range(len(res.editors)):
            item = res.editors[i]
            item.name = item.name.replace(' ','_').replace('.','_')
        return res

    perspectives = survey_subjects(example_topic)

    # [测试点3]: 检查角色生成是否正确
    
    pass


    # 专家对话
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    from langchain_core.messages import AnyMessage
    from typing import Annotated

    def add_messages(left, right):
        if not isinstance(left, list):
            left = [left]
        if not isinstance(right, list):
            right = [right]
        return left + right


    def update_references(references, new_references):
        if not references:
            references = {}
        references.update(new_references)
        return references


    def update_editor(editor, new_editor):
        # Can only set at the outset
        if not editor:
            return new_editor
        return editor


    class InterviewState(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]
        references: Annotated[Optional[dict], update_references]
        editor: Annotated[Optional[Editor], update_editor]

    # 多轮对话

    from langchain_core.prompts import MessagesPlaceholder
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

    # 独特角色生成问题的 提示词
    gen_qn_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
            """
            You are an experienced and multilingual writer and scholar, 
            and you wish to write an article on a specific topic. 
            In addition to your identity as a writer, you have a specific focus when researching the topic. 
            Now, you are chatting with an expert to gather information. 
            Ask good questions to obtain more useful information. 
            When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation. 
            Please ask only one question at a time and do not ask questions you have asked before. 
            Your questions should be related to the topic you want to write about. Be comprehensive and curious, gaining as much unique insight from the expert as possible. 
            Stay true to your specific perspective:

            {persona}""",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )

    def tag_with_name(ai_message: AIMessage, name: str):
        ai_message.name = name
        return ai_message


    def swap_roles(state: InterviewState, name: str):
        converted = []
        for message in state["messages"]:
            if isinstance(message, AIMessage) and message.name != name:
                message = HumanMessage(**message.dict(exclude={"type"}))
            converted.append(message)
        return {"messages": converted}

    # 从某个特定的角度提问
    # 人：专家
    # AI： 提出问题的 角色（Editor）
    def generate_question(state: InterviewState):
        editor = state["editor"]
        gn_chain = (
                RunnableLambda(swap_roles).bind(name=editor.name)
                | gen_qn_prompt.partial(persona=editor.persona)
                | gen_question_model
                | RunnableLambda(tag_with_name).bind(name=editor.name)
        )
        result = gn_chain.invoke(state)
        return {"messages": [result]}

    class Queries(BaseModel):
        queries: List[str] = Field(
            description="Comprehensive list of search engine queries to answer the user's questions.",
        )


    gen_queries_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant. Query the search engine to answer the user's questions.",
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )
    gen_queries_chain = gen_queries_prompt | create_question_model

    query_parser = PydanticOutputParser(pydantic_object=Queries)
    query_data_pattern = query_parser.get_format_instructions()

    # 总结 问题 提示词
    sum_queries_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant. Summarize below Queries:"
                "------"
                "Generate formatted data pattern as below:"
                "{data_pattern} ",
            ),
            ("user", "info: {info}"),
        ]
    )
    sum_queries_chain = sum_queries_prompt | sum_queries_model.with_structured_output(Queries, include_raw=True)

    # 回答问题
    def gen_answer(
            state: InterviewState,
            name: str = "Subject_Matter_Expert",
            max_str_len: int = 15000,
    ):
        swapped_state = swap_roles(state, name)

        # 生成初始问题
        res = gen_queries_chain.invoke(swapped_state)

        # 第一次为了得到AI 原始数据
        queries = sum_queries_chain.invoke({"info":res.content, "data_pattern": query_data_pattern})

        print('[产生问题节点] [{}]：-----------------------------'.format(state['editor']))
        global question_num
        global rlock
        local_index = 0

        # 第二次，使用注入方式再获得一次
        # 这里使用2次的原因是 某些模型 通过with_structured_output 解析出的内容，比通过query_parser 的方式要少...

        sec_sum_queries_chain = sum_queries_prompt | sum_queries_model | query_parser
        sec_queries = sec_sum_queries_chain.invoke({"info": res.content, "data_pattern": query_data_pattern})

        for q_item in sec_queries.queries:
            with rlock:
                question_num += 1
                local_index = question_num
                print('问题[{}]：{}\n'.format(local_index,q_item))
        print('--------------------------------------')
        # 默认只搜索一个问题

        query_results = doubao_search.batch(sec_queries.queries[:1], {"max_concurrency": 3})
        # 搜索所有问题
        #query_results = doubao_search.batch(sec_queries.queries,{"max_concurrency": 3})

        successful_results = [
            res for res in query_results if not isinstance(res, Exception)
        ]

        try:
            # 定义文件保存路径
            output_file_path = './query/successful_queries_[{}].txt'.format(local_index)

            # 打开文件准备写入
            with open(output_file_path, 'w', encoding='utf-8') as file:
                # 遍历成功的查询结果  默认只搜索一个
                for query, result in zip(sec_queries.queries[:1], successful_results):

                    # 进入临界区，保存数据
                    with rlock:
                        # 将访谈内容写到本地向量数据库
                        local_doc = Document(page_content=query + '\n\n' +result)
                        vector_store.add_documents([local_doc])

                    # 跳过异常结果
                    if isinstance(result, Exception):
                        continue

                    # 将问题和答案写入文件
                    file.write(f'问题: {query}\n')
                    file.write(f'答案: {result}\n')
                    file.write('-' * 80 + '\n')  # 用线分隔问题和答案

            print(f'成功结果已保存到 {output_file_path}')
        except Exception as e:
            print('问题和回答 保存异常 {}'.format(e))

        dumped = json.dumps(successful_results)[:max_str_len]
        ai_message: AIMessage = queries["raw"]

        if queries["raw"] is not None and len(queries["raw"].additional_kwargs)==0:
            swapped_state["messages"].extend([ai_message])
        else:
            tool_call = queries["raw"].additional_kwargs["tool_calls"][0]
            tool_id = tool_call["id"]
            tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
            swapped_state["messages"].extend([ai_message, tool_message])

        # 专家生成回答
        swapped_state['data_pattern'] =ans_citation_data_pattern
        generated_raw = gen_answer_chain.invoke(swapped_state)

        cited_references = {'茉卷知识库'}
        formatted_message = AIMessage(name=name, content=generated_raw.content)
        return {"messages": [formatted_message], "references": cited_references}
    class AnswerWithCitations(BaseModel):
        answer: str = Field(
            description="Comprehensive answer to the user's question with citations.",
        )
        cited_urls: List[str] = Field(
            description="List of urls cited in the answer.",
        )

        @property
        def as_str(self) -> str:
            return f"{self.answer}\n\nCitations:\n\n" + "\n".join(
                f"[{i + 1}]: {url}" for i, url in enumerate(self.cited_urls)
            )

    ans_citation_parser = PydanticOutputParser(pydantic_object=AnswerWithCitations)
    ans_citation_data_pattern = ans_citation_parser.get_format_instructions()
    # 专家回答问题 提示词
    gen_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert who can effectively utilize information. 
                You are conversing with a writer who wishes to write about a topic you are familiar with.
                You have gathered relevant information and will now use this information to form your responses. 
                Ensure that your responses are as informative as possible, and that every statement is supported by the gathered information. 
                Each response must be backed up by a citation from a reliable source, formatted as a footnote, with URLs provided after the response.
                ------
                Generate formatted data pattern as below:
                {data_pattern}  
                """,
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )

    gen_answer_chain = gen_answer_prompt | ans_questions_model

    max_num_turns = 5

    def route_messages(state: InterviewState, name: str = "Subject_Matter_Expert"):
        messages = state["messages"]
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        if num_responses >= max_num_turns:
            return END
        last_question = messages[-2]
        if last_question.content.endswith("Thank you so much for your help!"):
            return END
        return "ask_question"

    # 定义单个角色对话 Langgraph
    builder = StateGraph(InterviewState)
    builder.add_node("ask_question", generate_question)
    builder.add_node("answer_question", gen_answer)
    builder.add_conditional_edges("answer_question", route_messages)
    builder.add_edge("ask_question", "answer_question")

    builder.set_entry_point("ask_question")
    interview_graph = builder.compile().with_config(run_name="Conduct Interviews")
    final_step = None

    # 初步提纲
    outline_parser = PydanticOutputParser(pydantic_object=Outline)
    outline_parser_data_pattern = outline_parser.get_format_instructions()
    refine_outline_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a multilingual writer proficient in multiple languages. 
                You have gathered information from experts and search engines. 
                Now, you are refining the outline of your article. 
                You need to ensure that the outline is comprehensive and specific. 
                The topic you are writing about is: 
                {topic}. 
                Here is the old outline:
                {old_outline}
                """,
            ),
            (
                "user",
                """  
                Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\n
                ------
                Generate formatted data pattern as below:
                {data_pattern}  
                Write the refined outline:
                """
            ),
        ]
    )

    # 精炼提纲chain
    refine_outline_chain_pre = refine_outline_prompt | refine_outline_model | outline_parser


    # 多角色数组
    initial_states = [
        {
            "editor": editor,
            "messages": [
                AIMessage(
                    content=f"So you said you were writing an article on {example_topic}?",
                    name="Subject_Matter_Expert",
                )
            ],
        }
        for editor in perspectives.editors
    ]

    # 默认只执行一个角色
    interview_results = interview_graph.batch(initial_states[:1], {"max_concurrency": 2})

    # 全部执行
    #interview_results = interview_graph.batch(initial_states, {"max_concurrency": 2})

    # 将多个角色的对话历史合并
    talks = interview_results[0]['messages']
    for i in range(len(interview_results)):
        if i >= 1:
            talks = talks + interview_results[i]['messages'][1:]

    # 根据多角色对话历史 精炼 提纲
    refined_outline = refine_outline_chain_pre.invoke(
        {
            "topic": example_topic,
            "old_outline": initial_outline,
            "conversations": "\n\n".join(
                f"### {m.name}\n\n{m.content}" for m in talks
            ),
            "data_pattern":outline_parser_data_pattern
        }
    )

    # 详写各个段落
    def retrieve(inputs: dict):
        global rlock
        with rlock:
            docs = retriever.invoke(inputs["topic"] + ": " + inputs["section"])
        formatted = "\n".join(
            [
                f'<Document "/>\n{doc.page_content}\n</Document>'
                for doc in docs
            ]
        )
        return {"docs": formatted, **inputs}


    section_writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert writer. Complete your assigned work from the following outline:\n\n"
                "{outline}\n\nCite your sources, using the following references:\n\n<Documents>\n{docs}\n<Documents>",
            ),
            ("user", "Write the full content for the {section} section."),
        ]
    )


    class WorkSection(BaseModel):
        section_title: str = Field(..., title="Title of the section")
        content: str = Field(..., title="Full content of the section")
        subsections: Optional[List[Subsection]] = Field(
            default=None,
            title="Titles and descriptions for each subsection of the Wikipedia page.",
        )
        citations: List[str] = Field(default_factory=list)

        @property
        def as_str(self) -> str:
            subsections = "\n\n".join(
                subsection.as_str for subsection in self.subsections or []
            )
            citations = "\n".join([f" [{i}] {cit}" for i, cit in enumerate(self.citations)])
            return (
                    f"## {self.section_title}\n\n{self.content}\n\n{subsections}".strip()
                    + f"\n\n{citations}".strip()
            )

    # 内容书写chain: （1） 根据 topic 和section 从向量数据库获取数据 （2）内容书写
    section_writer = (
            retrieve
            | section_writer_prompt
            | section_writer_model
    )

    outline = refined_outline

    section_frtame = [
            {
                "outline": refined_outline.as_str,
                "section": section.section_title,
                "topic": example_topic,
            }
            for section in outline.sections
        ]

    # 各个段落内容生成
    sections = section_writer.batch(section_frtame,{"max_concurrency": 2})

    # 最终生成的文件
    file_path = './query/sections.txt'

    with open(file_path, 'w', encoding='utf-8') as file:
        for section_content in sections:
            file.write(section_content.content + '\n')
            file.write('\n')
    pass
    print('文章创建完成！')