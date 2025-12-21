from tavily import TavilyClient
from config import Settings
import trafilatura 

settings = Settings()

tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

class SearchService:
    def web_search(self ,query:str):
        results = []
        MAX_CHARS = 4000
        response = tavily_client.search(query,max_results=2)
        search_results = response.get("results",[])
        
        for result in search_results :
            downloaded = trafilatura.fetch_url(result.get("url"))
            content = trafilatura.extract(downloaded,include_comments=False)

            if not content:
                continue

            content = content[:MAX_CHARS]

            results.append({
                "title":result.get("title",""),
                "url" : result.get("url",""),
                "content":content
            })

        return results
