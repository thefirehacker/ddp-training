# What a Tensor Really Is

The video below, by Dan Fleisch, is not using the loose “deep learning slang” definition of tensor. It points at the more mathematical idea: a tensor is not important just because it can be written as a grid of numbers, but because it follows a specific rule when you change coordinates or transform the space you are working in. ([What’s a Tensor? — YouTube](https://www.youtube.com/watch?v=f5liqUk0ZTw))

That distinction matters, because students often learn the machine learning version first and come away thinking a tensor is just “anything with dimensions.” That is useful for coding, but incomplete.

This note cleanly separates the two meanings and then connects them.

**Related material in this repo:** the main walkthrough is [tutorial.md](tutorial.md); model tensors appear concretely in [`nanogpt-speedrun/src/train_gpt2.py`](../../nanogpt-speedrun/src/train_gpt2.py).

---

## The coding meaning

In PyTorch and most deep learning libraries, a tensor is the main data object used to store inputs, outputs, parameters, and intermediate activations. PyTorch describes tensors as a specialized data structure very similar to arrays and matrices, and they are the central abstraction used in models. ([PyTorch Tutorials — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html))

In that practical sense:

* a single number can be treated as a tensor,
* a list of numbers can be treated as a tensor,
* a table of numbers can be treated as a tensor,
* and higher-dimensional collections of numbers are also tensors. ([PyTorch Tutorials — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html))

This is the meaning students meet first in deep learning. An image might be stored as height × width × channels. A batch of token embeddings might be stored as batch × sequence length × hidden size. A tensor, in code, is the container that holds these values and lets the framework run fast numerical operations on them. ([PyTorch Tutorials — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html))

That definition is useful, but it is not the whole story.

---

## The mathematical meaning

Mathematically, a tensor is an object whose components change in a well-defined way when you change the basis or coordinates used to describe it. The same underlying tensor can be represented by different arrays depending on the coordinate system, but the object itself is not “the array”; the array is only its representation in a chosen basis. ([Wikipedia — Tensor](https://en.wikipedia.org/wiki/Tensor))

This is the core idea the video is trying to push students toward.

A vector is a good stepping stone. If you rotate your coordinate axes, the numbers you use to describe the vector change, but the vector itself has not magically become a different geometric object. The coordinates changed because your description system changed. A tensor generalizes this idea: its components also transform, but according to a specific rule associated with its type. ([Wikipedia — Tensor](https://en.wikipedia.org/wiki/Tensor))

So the important idea is this:

A tensor is not merely “a box of numbers.” It is a geometric or multilinear object that can be represented by numbers, and those numbers must transform consistently when the basis changes. ([Wikipedia — Tensor](https://en.wikipedia.org/wiki/Tensor))

---

## Why this feels confusing

The confusion happens because in machine learning we usually skip the geometric story and jump straight to implementation.

When you write code, you interact with `torch.Tensor`. That object behaves like a multidimensional array. PyTorch is not asking you, every time, whether your tensor is covariant, contravariant, or a multilinear map. It is giving you a practical structure for computation. ([PyTorch Tutorials — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html))

So students hear:

“tensor = multidimensional array”

and that becomes their entire mental model.

But the more precise statement is:

“In deep learning, tensors are implemented as multidimensional arrays because that is a convenient representation for computation. In mathematics, tensor means something deeper than just the storage layout.” ([Wikipedia — Tensor (machine learning)](https://en.wikipedia.org/wiki/Tensor_(machine_learning)))

That is the bridge the video is trying to build.

---

## A cleaner way to think about rank and dimension

Students often mix up two ideas:

* how many axes an array has,
* what kind of tensor something is mathematically.

In coding conversations, people casually say things like “a rank-3 tensor” to mean a 3-dimensional array. In more formal mathematics and physics, tensor order/type is about how the object transforms and how many slots it has as a multilinear object, not just how many array axes you wrote down. ([Wikipedia — Tensor (machine learning)](https://en.wikipedia.org/wiki/Tensor_(machine_learning)))

That is why the same word can feel slippery. In ML, “rank” often means number of dimensions in the stored array. In mathematics, the deeper meaning is about the tensor’s structure and transformation law.

For beginners, the safest way to hold both ideas is:

* in code, a tensor is usually a multidimensional numerical object,
* in math, a tensor is an object with transformation rules, and arrays are just one way to represent it. ([PyTorch Tutorials — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html))

---

## Why vectors and matrices are already part of the tensor story

A scalar, vector, and matrix are not separate from tensors in the broad modern sense. They are special cases.

A scalar can be viewed as the simplest tensor-like object. A vector is a first step into objects whose coordinates change when basis changes. A matrix can often represent a linear map, and in many contexts can also be treated as a tensor representation. What changes across these cases is not just the number of entries, but what kind of object those entries are describing. ([Wikipedia — Tensor](https://en.wikipedia.org/wiki/Tensor))

That is why the video spends time on vectors before tensors. It is building the idea that representation is not the same thing as the thing being represented.

---

## Why machine learning still uses the word “tensor”

Because neural networks are built from operations on structured numerical objects: vector operations, matrix multiplications, batched matrix multiplications, and higher-dimensional data flows. The word tensor survives because it is broad enough to cover all of these structured numerical forms, and because many operations in ML can be viewed through multilinear algebra. ([Wikipedia — Tensor (machine learning)](https://en.wikipedia.org/wiki/Tensor_(machine_learning)))

For example, an embedding batch in a transformer may have shape `(batch, sequence, hidden_dim)`. Attention scores may have shape `(batch, heads, sequence, sequence)`. These are stored as tensors because the framework needs a general object that can represent and manipulate all of them efficiently. ([PyTorch Tutorials — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html))

So in practice, tensors are the language of deep learning computation.

---

## The most useful intuition to leave with

A good student-friendly way to say it is this:

A tensor is a structured mathematical object that can be represented by an array of numbers. In machine learning, we usually focus on the array because that is what we compute with. In mathematics, we focus on the transformation rule because that is what tells us what kind of object it really is. ([Wikipedia — Tensor (machine learning)](https://en.wikipedia.org/wiki/Tensor_(machine_learning)))

That one sentence resolves most of the confusion.

---

## Final takeaway

If your students are coding, they can safely think:

“A tensor is the general-purpose numerical object used by deep learning libraries.”

If they are trying to understand the video at a deeper level, they should think:

“A tensor is not defined only by shape; it is defined by how its components transform under a change of basis.” ([PyTorch Tutorials — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html))

That is the real heart of the lesson.

---

## References

| Topic | Link |
|-------|------|
| Video (Dan Fleisch) | [What’s a Tensor?](https://www.youtube.com/watch?v=f5liqUk0ZTw) |
| PyTorch tensors | [Tutorials — Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) |
| Mathematical tensor | [Wikipedia — Tensor](https://en.wikipedia.org/wiki/Tensor) |
| ML usage of “tensor” | [Wikipedia — Tensor (machine learning)](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) |
