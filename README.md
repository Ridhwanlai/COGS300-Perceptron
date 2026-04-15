# COGS300-Perceptron
My personal project for COGS300

/TITLE Table of Contents

1. What Is a Perceptron?
2. Why I Built This
3. How a Perceptron Relates to a Biological Neuron
4. Circuit Architecture — How My Physical Build Works
5. Bill of Materials
6. Building the Circuit — Phase by Phase
7. TinkerCad Prototypes
8. Processing Visualization
9. Jupyter Notebook — Software Perceptron
10. Training the Perceptron by Hand
11. The Perceptron Learning Rule (worked exp?)
12. Limitations, (explain why XOR Is Impossible Here)
13. Reflections
14. References & Citations



### Why I Built This
>
>Hi, my name is Ridhwan! This was my final project (a perceptron) for COGS 300: Understanding and Designing Cognitive Systems, a course at UBC, where we built autonomous maze-solving robots and learned about machine learning (ML) topics for designing intelligent systems.
>
>I'm interested in healthcare and the development of transparent, interpretable AI systems. A recurring concern I have with modern ML is that we often deploy systems without truly understanding their internal mechanics, a form of epistemic dishonesty that carries real risk when those systems influence medical or safety-critical decisions.
>
>Building a perceptron physically—where every weight is a dial you can turn, every summation is a voltage you can probe with a multimeter, and the decision boundary is literally a wire connecting an op-amp to an LED—forces genuine mechanistic understanding. There is no abstraction to hide behind. Every concept from the theory (weighted sum, threshold, convergence) has a direct, tangible counterpart on the breadboard (that you could also see through an interface I generated using Claude in Processing).
>
>This project emerged from a sketch I showed my professor: a biological neuron I had drawn by hand, and the words "this is what I want to build." Well, that vision evolved into a perceptron, and after many late nights and numerous hours of fine-tuning, re-understanding concepts, and crashing out (every once in a while), here we are. Thank you to Paul for guiding me in the right direction and for being there to troubleshoot; it's moments like these that I'm reminded of the value of an education!
>
>
><img width="455" height="251" alt="IMG 5848" src="https://github.com/user-attachments/assets/f3172926-0194-439d-9ac6-701db2fa5c55" />
>
>*Figure 1: The 'Pitches and Sketches' Diagram that I showed Paul*




### How a Perceptron Relates to a Biological Neuron
>
>The perceptron is directly inspired by the structure of a biological neuron.
>
>In the brain, a neuron receives signals through its dendrites, integrates them in the cell body (soma), and—if the total signal crosses a threshold—fires an electrical spike down the axon, across the synaptic cleft, to the dendrites of the next neuron.
>
>| Biological Neuron                          | Analog Perceptron                          |
>| ------------------------------------------ | ------------------------------------------ |
>| Dendrites receiving signals                | DIP switch inputs x₁, x₂                   |
>| Synapse strength                           | Potentiometer dial position (w₁, w₂)       |
>| Axon hillock — summation + threshold check | TL074 op-amp summing node + comparator     |
>| Action potential (fires or doesn't)        | LED lights (green = fires, blue = doesn't) |
>| Axon → next neuron                         | Output wire → next circuit stage           |
>| Synaptic plasticity (learning)             | Turning the potentiometer dials by hand    |
>
When multiple perceptrons are combined—with the output of one feeding the inputs of others—you get a multi-layer neural network, capable of solving problems no single perceptron can. This is the architecture shown in the upper-left part of my notebook sketch (the multi-layer diagram with attention mechanisms), and the direction I intend for this project. 

// NEXT IMAGE

Figure 2: My full prototype, written portion...

Figure 3: a box of my protoype

My first iterations of this were a cardboard box with drawings, and the actual materials that I initially thought I'd be using. 


If you'd like to watch a video: https://drive.google.com/file/d/1gP8iWLKEKyuuV7yGvbxWcRd90X-XedA8/view 

<img width="1080" height="653" alt="Screenshot 2026-04-15 at 2 39 56 AM" src="https://github.com/user-attachments/assets/b2ca9573-be8e-407c-a2b5-5d4dc9c7fdf9" />

Figure 4: Image of Piazza post sharing initial plans for what used to be a Neural Network (way




Now for the part that you've been waiting for (drumroll please)... 

P.S. Thank you for following along so far, and if some of this stuff makes absolutely no sense, great! I was in your shoes a few months ago. :) 



/TITLE What Is a Perceptron?

A perceptron is the simplest possible form of an artificial neural network. The main idea was conceived by Frank Rosenblatt in 1957; it's the fundamental building block from which every modern deep learning model—GPT, Claude, AlphaFold—ultimately descends.

At its core, a perceptron does three things:
- Takes multiple inputs and assigns each a numerical weight
- Computes a weighted sum of those inputs plus a constant bias term
- Applies a threshold (activation) function — outputs 1 if the sum exceeds the threshold, 0 (or -1) if it does not

Mathematically, the output is:
ŷ = sign(w₁x₁ + w₂x₂ + b)

Where:
x₁, x₂ are the binary inputs (switches: ON = 1, OFF = 0 || (-1))
w₁, w₂ are the weights (set by the potentiometers)
b is the bias (a constant offset that shifts the decision boundary)
sign() is the activation function — positive sum → Class A, a zero, or negative sum → Class B

The perceptron is a binary linear classifier. Geometrically, the equation w₁x₁ + w₂x₂ + b = 0 defines a straight line in 2D space. Everything on one side of that line is Class A; everything on the other is Class B. Training the perceptron means adjusting w₁, w₂, and b until that line correctly separates all your training examples. [GeeksForGeeks: What Is a Perceptron?] (HAVE // THIS BE A LINK TO THE BOTTOM)

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/075e5531-0592-40f9-9329-bc18fefd3d61" />

Image Source: GeeksForGeeks — Perceptron diagram illustrating weighted inputs, summation, and activation function.


