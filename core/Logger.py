from colorama import Fore, Style, init

init(autoreset=True)
VERBOSE = 3

def printInfo(message: str) -> None:
    if VERBOSE >= 3:
        print(Fore.BLUE + "[Info]" + Style.RESET_ALL + f" - {message}")


def printWarning(message: str) -> None:
    if VERBOSE >= 2:
        print(Fore.YELLOW + "[Warning]" + Style.RESET_ALL + f" - {message}")


def printError(message: str) -> None:
    if VERBOSE >= 1:
        print(Fore.RED + "[Error]" + Style.RESET_ALL + f" - {message}")


if __name__ == "__main__":
    printInfo("info Test")
    printWarning("Warning Test")
    printError("Error Test")