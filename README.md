# DataFlow

Implementing neural networks in C++.

My PC is relatively slow and waiting for models to train in **keras/tensorflow** takes ages.
I decided to write everything from scratch in C++. 

Since I plan on calling C++ from Python, 
one can be sophisticated and actually write a Python script which places a runtime parameter 
in the source code and compiles the program, and then calls it from Python. This is a planned 
feature.

That way we get the best of both worlds. Python is king
when it comes to Data Science/prototyping/automatization, 
C++ takes the lead in raw performance.

## Things that make this fast:
- Data oriented (hense the "very original" name), and not object oriented, because I respect the CPU cache.
- Allocate all training batches close together in memory, because I respect the CPU cache.
- The architecture is known at compile-time, because that way the C++ compiler can do optimizations that otherwise wouldn't be possible. 
- Not even doing SIMD yet, but on the TODO list!

## In the future:
- There will be a script that runs a C++ compiler and then calls the .exe from Python seemlessly (e.g. from a jupyter notebook).
- I'll add a graphical window that pops up and shows plots in real-time that show how the model is training.
  - Once we have that up and running the potential is infinite.

## Contribution
If you want to help with this project, feel free to hit me up, open issues, do PRs.\
I (probably) won't accept PRs that add big features before I ensure they are programmed the way I want them to be.

PRs that add small features - e.g. bug fixes, more loss/activation functions, etc. are welcome! 

## Readiblity and Conciseness
I highly value code readiblity and code conciseness. This project uses a library of mine - **lstd**. A light replacement for the C++ standard library.

It's currently small but very powerful for the basic stuff - memory allocations, data structures, etc. I constantly add new 
features as I need them and the API is inspired by Python. Honestly it's the only thing that makes me still enjoy programming in this language.

Don't be scared of C++! Just because it's a low-level language doesn't mean the code should look terribly ugly. Just avoid any std::'s (like in real-life).
