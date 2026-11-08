import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handlers.ContextHandle import ContextHandle

history = [ContextHandle.Create("Jesteś wystrzałowym kotem")]
print(ContextHandle.Read(history, ["system"]))

history = ContextHandle.Add(history, "Jesteś Guźdzcem",1)
print(ContextHandle.Read(history, ["system"]))

history = ContextHandle.Add(history, "Jesteś Pumba",1)
print(ContextHandle.Read(history, ["system"]))

history = ContextHandle.Update(history, "Jesteś wystrzałowym typem")
print(ContextHandle.Read(history,["system"]))

history = ContextHandle.Update(history, "Jesteś PAN ŚWINIA", 2)
print(ContextHandle.Read(history,["system"]))