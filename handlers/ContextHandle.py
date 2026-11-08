class ContextHandle:

    @staticmethod
    def Create(prompt, role="system", **kwargs) -> dict:
        return {"role": role, "content":prompt, **kwargs}

    @staticmethod
    def Add(history: list, prompt, pos=None, role="system", **kwargs) -> list:
        message = ContextHandle.Create(prompt, role, **kwargs)

        if pos is None:
            history.append(message)
        else:
            history.insert(pos, message)

        return history
    
    @staticmethod
    def Append(history: list, message: dict) -> list:
        history.append(message)
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
