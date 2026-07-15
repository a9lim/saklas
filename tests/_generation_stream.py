"""Current-shape generation stream fixture."""

from collections.abc import Iterable, Iterator

from saklas.core.results import GenerationResult, TokenEvent


class TestGenerationStream(Iterator[TokenEvent]):
    __test__ = False

    def __init__(
        self,
        events: Iterable[TokenEvent],
        result: GenerationResult | None,
        error: BaseException | None = None,
    ) -> None:
        self._events = iter(events)
        self._result = result
        self.closed = False
        self._error = error

    def __iter__(self) -> "TestGenerationStream":
        return self

    def __next__(self) -> TokenEvent:
        try:
            return next(self._events)
        except StopIteration:
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            raise

    def close(self) -> None:
        self.closed = True

    @property
    def result(self) -> GenerationResult | None:
        return self._result
