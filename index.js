import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import dotenv from "dotenv";
dotenv.config();

// Initialize Gemini
const apiKey = process.env.GEMINI_API_KEY;

const llm = new ChatGoogleGenerativeAI({
  apiKey,
  model: "gemini-1.5-pro",
  temperature: 0,
  maxRetries: 2,
  // other params...
});

// Define tools
const multiply = tool(async ({ a, b }) => a * b, {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number().describe("first number"),
    b: z.number().describe("second number"),
  }),
});

const add = tool(async ({ a, b }) => a + b, {
  name: "add",
  description: "Add two numbers",
  schema: z.object({
    a: z.number().describe("first number"),
    b: z.number().describe("second number"),
  }),
});

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

const tools = [add, multiply, divide];
const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
const llmWithTools = llm.bindTools(tools);

// Modified llmCall to use direct Gemini API
async function llmCall(state) {
  const result = await llmWithTools.invoke([
    {
      role: "system",
      content:
        "You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
    },
    ...state.messages,
  ]);

  return {
    messages: [result],
  };
}

async function toolNode(state) {
  const results = [];
  const lastMessage = state.messages.at(-1);

  if (lastMessage?.tool_calls?.length) {
    for (const toolCall of lastMessage.tool_calls) {
      const tool = toolsByName[toolCall.name];
      const observation = await tool.invoke(toolCall.args);
      results.push({
        role: "tool",
        content: observation.toString(),
        tool_call_id: toolCall.id,
      });
    }
  }

  return { messages: results };
}

function shouldContinue(state) {
  const lastMessage = state.messages.at(-1);
  if (lastMessage?.tool_calls?.length) {
    return "Action";
  }
  return "__end__";
}

const agentBuilder = new StateGraph(MessagesAnnotation)
  .addNode("llmCall", llmCall)
  .addNode("tools", toolNode)
  .addEdge("__start__", "llmCall")
  .addConditionalEdges("llmCall", shouldContinue, {
    Action: "tools",
    __end__: "__end__",
  })
  .addEdge("tools", "llmCall")
  .compile();

// Example usage
const messages = [{
  role: "user",
  content: "Add 3 and 4."
}];
const result = await agentBuilder.invoke({ messages });
console.log(result.messages);
