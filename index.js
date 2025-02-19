import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {ChatOpenAI} from "@langchain/openai"
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";


const llm = new ChatOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o",
})

const multiply = tool(
  async ({ a, b }) => a * b,
  {
    name: "multiply",
    description: "Multiply two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const add = tool(
  async ({ a, b }) => a + b,
  {
    name: "add",
    description: "Add two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);


const divide = tool(
  async ({ a, b }) => {
    return a / b;
  },
  {
    name: "divide",
    description: "Divide first number by second number",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number (non-zero)"),
    }),
  }
);

const tools =[add, multiply, divide];
const toolsByName = Object.fromEntries(tools.map((tool)=>[tool.name, tool]));
const llmWithTools = llm.bindTools(tools);


async function llmCall(state){
  const result = await llmWithTools.invoke([
    {
      role: "system",
      content:"You are a helpful assistant asked with the prforming the arithematic on a set of input"
    },
    ...state.messages
  ])

  return{
    messages: [result],
  }
}


async function toolNode(state) {
  // Performs the tool call
  const results = [];
  const lastMessage = state.messages.at(-1);

  if (lastMessage?.tool_calls?.length) {
    for (const toolCall of lastMessage.tool_calls) {
      const tool = toolsByName[toolCall.name];
      const observation = await tool.invoke(toolCall.args);
      results.push(
        new ToolMessage({
          content: observation,
          tool_call_id: toolCall.id,
        })
      );
    }
  }

  return { messages: results };
}



function shouldContinue(state) {
  return "__end__";
}