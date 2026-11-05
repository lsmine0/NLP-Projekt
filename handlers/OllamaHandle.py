from typing import Any
from openai import OpenAI 
from handlers import ContextHandle as Contx
from handlers import ToolHandle as Thandle
import requests
from core import Logger


class OllamaHandler:
  def __init__(self, base_url="http://localhost:11434/v1", key="ollam"):
    if "://" not in base_url:
      base_url = f"http://{base_url}"
    self.base_url = base_url.rstrip("/").removesuffix("/v1") + "/v1"
    self.client = OpenAI(
    base_url=self.base_url,
    api_key=key
    )
    self.model = None
    self.thinking = None
    self.ToolsHandle = Thandle.ToolHandle()

  def GetModels(self):
    Logger.printInfo("Getting model list.")
    return self.client.models.list()

  def SetModel(self,ModelName):
    if ModelName in [model.id for model in self.GetModels()]:
      self.model = ModelName
      Logger.printInfo(f"Model {ModelName} set on endpoint.")
      return True
    else: 
      Logger.printWarning(f"model {ModelName} not avaible on this Ollama endpoint")
      return False

  def GetModelInfo(self):
      if self.model:

        response = requests.post(
            f"{self.base_url.removesuffix('/v1')}/api/show",
            json={"model": self.model},
            timeout=10,
        )
        response.raise_for_status()
        Logger.printInfo(f"Model {self.model} Getinfo.")
        return response.json()
      else:
        Logger.printError(f"Get info on Invalid model.")
        return {}

  def GetCapabilities(self):
    return self.GetModelInfo().get("capabilities", [])      
  
  def GetThinking(self):
    return self.GetModelInfo().get("thinking")

  def SetThinking(self, level):
        thinking_info = self.GetThinking()

        if thinking_info is None:
            if level is not None:
                Logger.printError(f"Model does not support thinking.")
                return
            self.thinking = None
            return

        supported = thinking_info["values"]

        if level not in supported and level is not None:
            Logger.printWarning(f"Unsupported thinking level.")

        self.thinking = level

  def GetVision(self):
    return "vision" in self.GetCapabilities()         

  # Przygotowanie parametrów wspólnych dla Chat i ChatStream
  def _Prepare(self, kwargs: dict[str, Any]):
    if self.model is None:
        Logger.printError(f"No model selected")
        raise RuntimeError("No model selected")

    extra_body = dict(kwargs.pop("extra_body", {}) or {})

    if self.thinking is not None:
        extra_body["think"] = self.thinking

    tools = self.ToolsHandle.get_tools_list()
    if tools:
        kwargs["tools"] = tools

    Logger.printInfo(f"Sending request to {self.model}")
    return self.model, (extra_body or None), kwargs

  def Chat(self, messages, **kwargs):
    model, extra_body, kwargs = self._Prepare(kwargs)
    return self.client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        extra_body=extra_body,
        **kwargs,
    )

  def ChatStream(self, messages, **kwargs):
    model, extra_body, kwargs = self._Prepare(kwargs)
    return self.client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        extra_body=extra_body,
        **kwargs,
    )
