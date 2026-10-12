# `gko::stop::Iteration`

Stops the solver after a fixed number of iterations. Use as a safety
net inside [`Combined`](combined.md) so a divergent solve cannot run
indefinitely.

`gko::stop::max_iters(n)` builds this criterion directly.
`gko::stop::min_iters(n, criterion)` wraps other criteria so that they are
only checked once `n` iterations have finished; with several criteria it
combines them first, and from then on it behaves like the wrapped
criteria.

```{doxygenclass} gko::stop::Iteration
:members:
```

```{doxygenfunction} gko::stop::max_iters
```

The two `min_iters` overloads are declared in `ginkgo/core/stop/iteration.hpp` as:

```text
deferred_factory_parameter<const CriterionFactory> min_iters(
    size_type count,
    deferred_factory_parameter<const CriterionFactory> criterion);

template <typename... Args>
std::enable_if_t<sizeof...(Args) >= 2,
                 deferred_factory_parameter<const CriterionFactory>>
min_iters(size_type count, Args&&... criteria);
```

`count` is the number of iterations after which checking of the wrapped
criteria starts. With two or more criteria, they are combined with
`Combined` (stop when any fires) inside the wrapper.
