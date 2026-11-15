"""Asynchroniczny runtime MCP z synchronicznym interfejsem dla reszty aplikacji."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, MCPError, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from core import Logger


class MCPErrorBase(RuntimeError):
    """Bazowy błąd warstwy MCP."""


class MCPConfigError(MCPErrorBase):
    """Nieprawidłowa konfiguracja serwera MCP."""


class MCPConnectionError(MCPErrorBase):
    """Błąd połączenia lub transportu MCP."""


class MCPTimeout(MCPErrorBase):
    """Przekroczono limit czasu operacji MCP."""


class MCPToolError(MCPErrorBase):
    """Serwer MCP zwrócił błąd narzędzia."""


@dataclass
class _Request:
    tool: str
    arguments: dict[str, Any]
    timeout: float
    future: concurrent.futures.Future


class MCPRuntime:
    """Utrzymuje jeden wątek asyncio i po jednym workerze na serwer MCP.

    Cała komunikacja z serwerem odbywa się w wątku event loopa. Kod
    synchroniczny (AgentFlow, ToolHandle) wysyła żądania do kolejki i czeka
    na ``concurrent.futures.Future``.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="mcp-runtime",
            daemon=True,
        )
        self._loop_started = threading.Event()
        self._lock = threading.RLock()
        self._closed = False
        self._queues: dict[str, asyncio.Queue] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._dead: dict[str, BaseException] = {}
        self._intentional_stop: set[str] = set()

    # ------------------------------------------------------------------
    # Cykl życia
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise MCPConnectionError("Runtime MCP został już zamknięty.")
            if self._thread.is_alive():
                return
            self._thread.start()

        if not self._loop_started.wait(5.0):
            raise MCPConnectionError("Nie udało się uruchomić wątku MCP.")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_started.set()
        try:
            self._loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

        if not self._thread.is_alive():
            try:
                self._loop.close()
            except RuntimeError:
                pass
            return

        async def _shutdown() -> None:
            for queue in list(self._queues.values()):
                try:
                    queue.put_nowait(None)
                except Exception:
                    pass

            tasks = [task for task in self._tasks.values() if not task.done()]
            if not tasks:
                return

            done, pending = await asyncio.wait(tasks, timeout=3.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            future.result(timeout=5.0)
        except Exception:
            pass
        finally:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Połączenie z serwerem
    # ------------------------------------------------------------------
    def connect(
        self,
        name: str,
        params: StdioServerParameters,
        timeout: float,
    ) -> tuple[Any, list[Any]]:
        """Uruchamia serwer, wykonuje handshake i zwraca ``(init, tools)``."""

        self.start()

        with self._lock:
            task = self._tasks.get(name)
            if task is not None and not task.done() and name not in self._dead:
                raise MCPConnectionError(f"Serwer MCP '{name}' jest już połączony.")

            self._dead.pop(name, None)
            self._intentional_stop.discard(name)
            ready = concurrent.futures.Future()
            self._loop.call_soon_threadsafe(self._spawn_worker, name, params, ready)

        try:
            return ready.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self._request_stop(name)
            with self._lock:
                task = self._tasks.get(name)
            if task is not None and not task.done():
                try:
                    self._loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    pass
            raise MCPTimeout(
                f"Serwer MCP '{name}' nie odpowiedział w ciągu {timeout:.1f} s."
            )

    def _spawn_worker(
        self,
        name: str,
        params: StdioServerParameters,
        ready: concurrent.futures.Future,
    ) -> None:
        if self._closed:
            if not ready.done():
                ready.set_exception(MCPConnectionError("Runtime MCP jest zamknięty."))
            return

        queue: asyncio.Queue = asyncio.Queue()
        self._queues[name] = queue
        task = self._loop.create_task(
            self._worker(name, params, queue, ready),
            name=f"mcp-{name}",
        )
        self._tasks[name] = task
        task.add_done_callback(lambda finished: self._on_worker_done(name, finished, queue, ready))

    def _on_worker_done(
        self,
        name: str,
        task: asyncio.Task,
        queue: asyncio.Queue,
        ready: concurrent.futures.Future,
    ) -> None:
        if task.cancelled():
            error: BaseException = MCPConnectionError(
                f"Połączenie MCP '{name}' zostało anulowane."
            )
        else:
            task_error = task.exception()
            error = self._wrap_error(task_error, name) if task_error is not None else MCPConnectionError(
                f"Połączenie MCP '{name}' zakończyło się."
            )

        if not ready.done():
            ready.set_exception(error)

        while True:
            try:
                request = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if request is None or request.future.done():
                continue
            request.future.set_exception(error)

        if name in self._intentional_stop:
            self._intentional_stop.discard(name)
        else:
            self._dead[name] = error

    async def _worker(
        self,
        name: str,
        params: StdioServerParameters,
        queue: asyncio.Queue,
        ready: concurrent.futures.Future,
    ) -> None:
        errlog = None
        try:
            errlog = self._open_errlog(name)
            async with AsyncExitStack() as stack:
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(params, errlog=errlog)
                )
                # Rodzic nie musi trzymać końca zapisu; dziecko ma własny deskryptor.
                try:
                    errlog.close()
                except Exception:
                    pass
                errlog = None

                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                init_result = await session.initialize()
                tools = await self._list_tools(session)

                if not ready.done():
                    ready.set_result((init_result, tools))

                while True:
                    request = await queue.get()
                    if request is None:
                        break
                    if request.future.cancelled():
                        continue

                    try:
                        result = await asyncio.wait_for(
                            session.call_tool(
                                request.tool,
                                request.arguments,
                                read_timeout_seconds=request.timeout,
                            ),
                            timeout=max(request.timeout + 1.0, 1.0),
                        )
                    except asyncio.TimeoutError:
                        if not request.future.done():
                            request.future.set_exception(
                                MCPTimeout(
                                    f"Narzędzie MCP '{name}/{request.tool}' przekroczyło "
                                    f"limit czasu ({request.timeout:.1f} s)."
                                )
                            )
                        continue
                    except MCPError as exc:
                        # Błąd JSON-RPC pojedynczego wywołania. Część kodów
                        # oznacza zerwane połączenie i wymaga restartu workera.
                        connection_error = self._is_connection_mcp_error(exc)
                        if not request.future.done():
                            if connection_error:
                                request.future.set_exception(
                                    MCPConnectionError(
                                        f"Połączenie MCP '{name}' zostało zerwane: "
                                        f"{exc.message}."
                                    )
                                )
                            elif self._is_timeout_mcp_error(exc):
                                request.future.set_exception(
                                    MCPTimeout(
                                        f"Narzędzie MCP '{name}/{request.tool}' przekroczyło "
                                        f"limit czasu ({request.timeout:.1f} s)."
                                    )
                                )
                            else:
                                request.future.set_exception(
                                    MCPToolError(
                                        f"Błąd MCP '{name}/{request.tool}': {exc.message}"
                                        f" (kod {exc.code})."
                                    )
                                )
                        if connection_error:
                            raise MCPConnectionError(
                                f"Połączenie MCP '{name}' zostało zerwane: {exc.message}."
                            )
                        continue
                    except Exception as exc:
                        if not request.future.done():
                            request.future.set_exception(self._wrap_error(exc, name))
                        raise

                    if not request.future.done():
                        request.future.set_result(result)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not ready.done():
                ready.set_exception(self._wrap_error(exc, name))
            raise
        finally:
            if errlog is not None:
                try:
                    errlog.close()
                except Exception:
                    pass

    async def _list_tools(self, session: ClientSession) -> list[Any]:
        tools: list[Any] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()

        for _ in range(100):
            params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
            result = await session.list_tools(params=params)
            tools.extend(result.tools)
            cursor = result.next_cursor
            if not cursor:
                return tools
            if cursor in seen_cursors:
                raise MCPConnectionError("Serwer MCP zapętlił paginację tools/list.")
            seen_cursors.add(cursor)

        raise MCPConnectionError("Serwer MCP zwrócił zbyt wiele stron tools/list.")

    # ------------------------------------------------------------------
    # Wywołania
    # ------------------------------------------------------------------
    def call_tool(
        self,
        name: str,
        tool: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> Any:
        with self._lock:
            if self._closed:
                raise MCPConnectionError("Runtime MCP jest zamknięty.")
            task = self._tasks.get(name)
            queue = self._queues.get(name)
            if queue is None or task is None or task.done() or name in self._dead:
                raise MCPConnectionError(f"Serwer MCP '{name}' nie jest połączony.")

        request = _Request(
            tool=tool,
            arguments=arguments,
            timeout=timeout,
            future=concurrent.futures.Future(),
        )
        try:
            self._loop.call_soon_threadsafe(queue.put_nowait, request)
        except RuntimeError as exc:
            raise MCPConnectionError(f"Runtime MCP '{name}' nie działa.") from exc

        try:
            return request.future.result(timeout=max(timeout + 2.0, 2.0))
        except concurrent.futures.TimeoutError:
            request.future.cancel()
            raise MCPTimeout(
                f"Narzędzie MCP '{name}/{tool}' przekroczyło limit czasu ({timeout:.1f} s)."
            )

    def restart(
        self,
        name: str,
        params: StdioServerParameters,
        timeout: float,
    ) -> tuple[Any, list[Any]]:
        self._request_stop(name)
        self._wait_for_stop(name, timeout=min(timeout, 5.0))
        return self.connect(name, params, timeout)

    def is_alive(self, name: str) -> bool:
        with self._lock:
            task = self._tasks.get(name)
            return task is not None and not task.done() and name not in self._dead

    # ------------------------------------------------------------------
    # Zatrzymywanie pojedynczego serwera
    # ------------------------------------------------------------------
    def _request_stop(self, name: str) -> None:
        with self._lock:
            queue = self._queues.get(name)
            self._intentional_stop.add(name)
            self._dead.pop(name, None)

        if queue is not None:
            try:
                self._loop.call_soon_threadsafe(queue.put_nowait, None)
            except RuntimeError:
                pass

    def _wait_for_stop(self, name: str, timeout: float) -> None:
        with self._lock:
            task = self._tasks.get(name)
        if task is None or task.done():
            return

        done = threading.Event()

        def _watch() -> None:
            if task.done():
                done.set()
            else:
                task.add_done_callback(lambda _: done.set())

        try:
            self._loop.call_soon_threadsafe(_watch)
        except RuntimeError:
            return

        if done.wait(timeout):
            return

        try:
            self._loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            return
        done.wait(2.0)

    # ------------------------------------------------------------------
    # Pomocnicze
    # ------------------------------------------------------------------
    def _wrap_error(self, error: BaseException | None, name: str) -> BaseException:
        if isinstance(error, (MCPConnectionError, MCPTimeout, MCPToolError)):
            return error
        return MCPConnectionError(f"Błąd połączenia MCP '{name}': {error}")

    @staticmethod
    def _is_connection_mcp_error(error: MCPError) -> bool:
        message = (error.message or "").lower()
        return error.code == -32000 and (
            "connection" in message
            or "closed" in message
            or "broken" in message
            or "eof" in message
        )

    @staticmethod
    def _is_timeout_mcp_error(error: MCPError) -> bool:
        message = (error.message or "").lower()
        return error.code == -32001 or "timed out" in message or "timeout" in message

    def _open_errlog(self, name: str) -> Any:
        read_fd, write_fd = os.pipe()
        thread = threading.Thread(
            target=self._drain_stderr,
            args=(name, read_fd),
            name=f"mcp-stderr-{name}",
            daemon=True,
        )
        thread.start()
        return os.fdopen(write_fd, "w", encoding="utf-8", errors="replace", buffering=1)

    @staticmethod
    def _drain_stderr(name: str, read_fd: int) -> None:
        try:
            with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    line = line.rstrip()
                    if line:
                        Logger.printInfo(f"[mcp:{name}] {line}")
        except Exception:
            pass
