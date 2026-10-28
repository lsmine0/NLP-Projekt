class ContextHandle:

    @staticmethod
    def Create(prompt, role="system") -> dict:
        return {"role": role, "content":prompt}

    @staticmethod
    def Add(history:list, prompt,pos=0) -> list:
        history.insert(pos,ContextHandle.Create(prompt))
        return history

    @staticmethod
    def Update(history:list, prompt,pos=0) -> list:
        history[pos] = ContextHandle.Create(prompt) 
        return history

    @staticmethod
    def Read(history, lista)-> str:
        wybrane = ""
        for wiad in history:
            if wiad["role"] in lista:
                wybrane = wybrane + wiad["role"] + " : " + wiad["content"] + "\n"
        return wybrane
