# COGS300-Perceptron
My personal project for COGS300

/TITLE Table of Contents

1. Introduction and Background
- Why I Built This
- How a Perceptron Relates to a Biological Neuron

Perceptron Breakdown
- What Is a Perceptron?
- Circuit Architecture — How my Physical Build Works
- Processing Visualization (Generation Aided by Claude)
- Jupyter Notebook — Software Perceptron
- Limitations — Why XOR is Impossible (for my Perceptron)

Conclusion
- Challenges - Explained through TinkerCad Prototypes
- Reflections (and Takeaways)
- References & Citations





## Introduction and Background

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
>The perceptron is directly inspired by the structure of a biological neuron.[^1]
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
>*Figure 2: A table showing the comparison between a biological neuron and an analog perceptron*
>
>When multiple perceptrons are combined—with the output of one feeding the inputs of others—you get a multi-layer neural network, capable of solving problems no single perceptron can. This is the architecture shown in the upper-left part of my notebook sketch (the multi-layer diagram with attention mechanisms), and the direction I intended for this project. 
>
><img width="381" height="285" alt="Image from iLoveIMG" src="https://github.com/user-attachments/assets/e56e0793-2e38-409e-9bd8-60845f1fdf12" />
>
>*Figure 3: My perceptron prototype theory on paper*
>
>My first iterations of this were on a cardboard box (as my base) with physical components that I planned to use further down the line (perceptron, switches, etc.) If you'd like to watch a video: https://drive.google.com/file/d/1gP8iWLKEKyuuV7yGvbxWcRd90X-XedA8/view
>
><img width="387" height="354" alt="BIG IMAGE" src="https://github.com/user-attachments/assets/a0f5ca0f-d4e3-4756-9257-ea23c33b1a83" />
>
>*Figure 4: My actual work-like perceptron paper prototype*
>
><img width="540" height="327" alt="Screenshot 2026-04-15 at 2 39 56 AM" src="https://github.com/user-attachments/assets/b2ca9573-be8e-407c-a2b5-5d4dc9c7fdf9" />
>
> *Figure 5: Image of my Piazza post sharing the math behind the first layer of my neural network*

Now for the part that you've been waiting for (drumroll please)... Before that though, thank you for following along so far, and if some of this stuff makes absolutely no sense, great! I was in your shoes a few months ago. :) 


## Perceptron Breakdown

### What Is a Perceptron?
>
> A perceptron is the simplest possible form of an artificial neural network. The main idea was conceived by Frank Rosenblatt in 1957; it's the fundamental building block from which every modern deep learning model—GPT, Claude, AlphaFold—ultimately descends.[^1]
>
> **At its core, a perceptron does three things:**
> * Takes multiple inputs and assigns each a numerical weight
> * Computes a weighted sum of those inputs plus a constant bias term
> * Applies a threshold (activation) function — outputs 1 if the sum exceeds the threshold, 0 (or -1) if it does not
>   
> Mathematically, the output is: $ŷ = \sum_{i=1}^{n}(w₁x₁ + w₂x₂ + b)$
>
> **Where:**
> * $x₁$, $x₂$ are the binary inputs (switches: ON = 1, OFF = 0 || (-1))
> * $w₁$, $w₂$ are the weights (set by the potentiometers)
> * $b$ is the bias (a constant offset that shifts the decision boundary)
> * $\sum_{i=1}^{n} ()$ is the activation function - positive sum → Class A, a zero, or negative sum → Class B
> 
> The perceptron is a binary linear classifier. Geometrically, the equation $w₁x₁ + w₂x₂ + b = 0$ defines a straight line in 2D space. Everything on one side of that line is Class A; everything on the other is Class B. Training the perceptron means adjusting $w₁$, $w₂$, and $b$ until that line correctly separates all your training examples.[^2]
>
> <img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/075e5531-0592-40f9-9329-bc18fefd3d61" />
>
> *Figure 6: GeeksForGeeks - Perceptron diagram illustrating weighted inputs, summation, and activation function*



(organize this afterwards)
Circuit Architecture — How my Physical Build Works
The circuit implements the perceptron equation ŷ = sign(w₁x₁ + w₂x₂ + b) in five distinct physical stages:
Stage 1 — Inputs (x₁, x₂)
Two positions of a DIP switch act as binary inputs. When a switch is ON, it connects 9V to the corresponding weight potentiometer — representing input = +1. When OFF, a pull-down resistor holds the line firmly at 0V — representing input = −1. The pull-down is essential: without it, an open switch leaves the input floating at an undefined voltage, producing garbage readings.
Stage 2 — Weights (w₁, w₂, b)
Three 3386MP trimmer potentiometers act as the learnable weights. Each pot is wired as a voltage divider:

Left pin → input signal (from switch, or 9V directly for bias)
Right pin → GND
Center pin (wiper) → outputs a voltage proportional to how far the knob is turned

Turning the knob clockwise increases the weight (more voltage passes through). Counterclockwise approaches zero. The third pot (RV3) is always connected to 9V regardless of switch state — this is the bias term b, which shifts the decision boundary away from the origin.
Stage 3 — Summation Node (Σ)
The center (wiper) pins of all three pots connect to the same row on the breadboard through 220Ω series resistors. Because breadboard rows are internally connected, the three voltages combine passively. The voltage at this shared row is:
V_sum = (w₁x₁ + w₂x₂ + b) / 3
The division by 3 occurs because three equal resistors share the summing row — this is the resistive summing network. The relative relationships between inputs are preserved perfectly.

Your multimeter IS the output reader here. Set it to DC Volts. Red probe on the summing row, black probe on GND. The voltage you read is ŷ. Above 4.5V = positive prediction. Below 4.5V = negative prediction.

Stage 4 — Threshold / Activation Function (TL074)
The TL074 quad op-amp is used as a comparator — comparing the summing node voltage against a fixed reference voltage (Vref = 4.5V, set by a pair of equal 220Ω resistors forming a voltage divider from 9V to GND). [^4]
Two internal amplifiers of the TL074 are used:

Comparator 1 (pins 2, 3, 1): +IN = V_sum, −IN = Vref → output HIGH when sum > Vref → green LED
Comparator 2 (pins 5, 6, 7): +IN = Vref, −IN = V_sum → output HIGH when Vref > sum → blue LED [^5]

This gives mutually exclusive LED outputs — exactly one lights at a time.
Stage 5 — Output (ŷ)
Each op-amp output drives one LED through a 220Ω current-limiting resistor (without this, the LED draws too much current and burns out immediately).

Green LED = positive prediction = Class A
Blue LED = negative prediction = Class B





### Processing Visualization (Generation Aided by Claude)
>
> To make the internal behavior of the circuit visible and legible for a presentation audience, a companion Processing sketch was developed (heavily assisted by Cluade) that communicates with an Arduino over serial. **This can be found as one of the files within the repo.**
> 
> **The Arduino reads:**
> * The voltage at the summing node (Row S) via an analog input pin
> * The state of the two DIP switch inputs
> * It then streams this data as CSV over serial $(x1, x2, v_sum)$ to a Processing sketch running on a connected laptop
>   
> **The Processing visualization displays:**
> * A real-time dial/gauge showing the current value of $v_sum$ — styled with a CRT aesthetic, sweeping left to right as you turn the potentiometer knobs. This makes the summation visible in a way that a multimeter alone cannot convey to an audience watching from a distance.
> * A 2D decision boundary plot showing the current classification line $w₁x₁ + w₂x₂ + b = 0$, updating live as the pots are turned
> * LED indicators mirroring the physical LEDs, so the audience can see the binary output clearly on-screen
> * The current values of $w₁, w₂$, and $b$ displayed numerically as the knobs are adjusted
> This visualization transforms the physical circuit from a black box into a transparent, live-updating window into the perceptron's decision-making process, which is directly relevant to my interest in interpretable AI systems.
> ***Voltage safety note** if you're interested in replicating: The Arduino's analog inputs operate at 5V maximum. The op-amp summing output (0–9V) was stepped down via a 2:1 voltage divider (two equal resistors) before entering the analog pin. This prevents damage to the Arduino.*

### Jupyter Notebook — Software Perceptron
>
> To deepen my understanding of the perceptron learning algorithm and go beyond what a physical single-layer analog circuit can do, a software implementation was built in JupyterLab, copying an YouTube demonstration by ethicalPap's Perceptron Algorithm Learning series.[^3]
> 
> **The notebook implements:**
> * The perceptron learning rule from scratch in Python (no ML libraries)
> * Visualization of the decision boundary updating over training epochs
> * Training on the AND, OR, and NAND logic gate problems
> * A demonstration of why XOR fails (not linearly separable) — the motivation for multi-layer networks

### Limitations — Why XOR is Impossible (for my Perceptron)
>
> The fundamental limitation of a single-layer perceptron is that it can only solve linearly separable problems; problems where a single straight line can divide Class A from Class B. The XOR problem is the canonical example of a problem that is not linearly separable:
>
>
>| $x₁$ | $x₂$ | XOR output |
>| ---- | ---- | ---------- |
>| 0    | 0    | 0          |
>| 0    | 1    | 1          |
>| 1    | 0    | 1          |
>| 1    | 1    | 0          |
>
> No single straight line can separate the 1 outputs from the 0 outputs, as they form an alternating checkerboard pattern. This is why XOR was the problem that broke the first perceptron hype cycle in the 1960s.[^6]
>
> The solution is multiple layers (the multi-layer perceptron (MLP) shown in the upper-right of my original notebook sketch; figure 3). With a hidden layer, the network can learn non-linear boundaries by combining multiple linear decisions. This is the architecture behind every modern deep learning model, and the direction this project is intended to grow. My goal is to work on ensuring that my 'hidden layer' is still interpretable, and understanding the concepts within this project has gone a long way towards ensuring that.


## Conclusion

### Challenges
> TinkerCad Prototypes
>
> Before committing to the physical build, three full circuit simulations were developed in TinkerCad Circuits to validate the design and catch wiring errors before they became real problems. Tinkercad allows you to run the simulation, probe voltages, and flip switches in a virtual environment, which were invaluable for debugging a circuit this interconnected. It also made approaching some of the challenges I ran into much more easy to work through, because I could play around with a few different versions, to really try to get to the core problem.
>
> 
> Prototype 1 — Basic summing network (op-amp working) Validated that my three potentiometers wired to a shared row through equal resistors were in fact producing a voltage proportional to the weighted sum of the inputs. I struggled with summation here and my nodes were not directly feeding into the LED.
>
> upload image
>
> Prototype 2 — Single comparator with fixed threshold, including having a fully operational op-amp comparator stage with a fixed 4.5V Vref divider. Was able to confirm mutually exclusive LED behavior, exactly one LED on at a time, cleanly switching as the summing node crossed the threshold. The problem here though was my summation was actually subtracting from the total value of the row, which meant it never positively crossed the threshold.
>
> upload image
>
> Prototype 3 — Full circuit with dual comparators and DIP switch inputs Complete design: two-position DIP switch inputs with pull-down resistors, three pots (two for weights, one for bias), resistive summing network, dual TL074 comparator outputs, and both LEDs with current-limiting resistors. This is the final design that made me feel comfortable enough to transfer what I knew onto a physical breadboard.
>
> upload image
> 

### xx
>
>

### References & Citations
>
> [^1]: Rosenblatt, F. (1958). *The Perceptron: A probabilistic model for information storage and organization in the brain.* Psychological Review, 65(6), 386–408.
> [^2]: [GeeksForGeeks — What Is a Perceptron?](https://www.geeksforgeeks.org/deep-learning/what-is-perceptron-the-simplest-artificial-neural-network/)
> [^3]: [ethicalPap — Machine Learning 101, Lecture 2: Perceptron Algorithm](https://www.youtube.com/watch?v=ziDrxd_JTvs)
> [^4]: [Rohm Semiconductor — BA10324A Datasheet](https://media.digikey.com/pdf/Data%20Sheets/Rohm%20PDFs/BA10324A_AF_AFV.pdf)
> [^5]: [Texas Instruments — TL074 Datasheet](https://www.ti.com/product/TL074)
> [^6]: Minsky, M. & Papert, S. (1969). *Perceptrons.* MIT Press.




