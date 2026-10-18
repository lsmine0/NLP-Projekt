class ContextHandle:

    def Create(prompt) -> dict:
        return {"role": "system", "content":prompt}

    def Add(history:list, prompt,pos=0) -> list:
        history.insert(pos,ContextHandle.Create(prompt))
        return history

    def Update(history:list, prompt,pos=0) -> list:
        history[pos] = ContextHandle.Create(prompt) 
        return history

    def Read(history, lista)-> str:
        wybrane = ""
        for wiad in history:
            if wiad["role"] in lista:
                wybrane = wybrane + wiad["role"] + " : " + wiad["content"] + "\n"
        return wybrane