from openai import OpenAI 
from colorama import Fore, Back, Style

class OllamaHandler:
  def __init__(self, apiendpoint, key, verbose = 0):
    client = OpenAI(
    base_url = apiendpoint,
    api_key=key
    )
    model = None
    verbose = verbose

  def GetModels()
    return client.models.list()

  def SetModel(ModelName)
    if ModelName in GetModels():
      model = ModelName
      print(Fore.Blue + f"[Info]" + Style.RESET_ALL + "- model {ModelName} not avaible on this Ollama endpoint")
      return True
    else: 
      if verbose > 0:
        print(Fore.Red + "[Error]" + Style.RESET_ALL +" - model {ModelName} not avaible on this Ollama endpoint")
      return False
