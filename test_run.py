import asyncio 
from backend.agents.search_agent import decompose_query 
async def main(): 
    result = await decompose_query('what is a transformer', 2) 
    print(result) 
asyncio.run(main()) 
