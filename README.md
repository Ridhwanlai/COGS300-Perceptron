# Ridhwanlai's Perceptron

## Table of Contents

1. [Introduction and Background](#introduction-and-background)
    1. [Why I Built This](#why-i-built-this)
    2. [How a Perceptron Relates to a Biological Neuron](#how-a-perceptron-relates-to-a-biological-neuron)

2. [Perceptron Breakdown](#perceptron-breakdown)
    1. [What Is a Perceptron?](#what-is-a-perceptron)
    2. [Circuit Architecture — How My Physical Build Works](#circuit-architecture--how-my-physical-build-works)
    3. [Processing Visualization](#processing-visualization)
    4. [Jupyter Notebook — Software Perceptron](#jupyter-notebook--software-perceptron)
    5. [Limitations — Why XOR is Impossible (for my Perceptron)](#limitations--why-xor-is-impossible-for-my-perceptron)

3. [Conclusion](#conclusion)
    1. [Challenges - Explained through TinkerCad Prototypes](#challenges)
    2. [Reflections](#reflections)
    3. [References & Citations](#references--citations)

## Introduction and Background

### Why I Built This
>
>Hi, my name is Ridhwan! This was my final project (a perceptron) for COGS 300: Understanding and Designing Cognitive Systems, a course at UBC where we built autonomous maze-solving robots and learned about machine learning (ML) topics for designing intelligent systems.
>
>I'm interested in healthcare and the development of transparent, interpretable AI systems. A recurring concern I have with modern ML is that we often deploy systems without truly understanding their internal mechanics, a form of epistemic dishonesty that carries real risk when those systems influence medical or safety-critical decisions.
>
>Building a perceptron physically—where every weight is a dial you can turn, every summation is a voltage you can probe with a multimeter, and the decision boundary is literally a wire connecting an op-amp to an LED—forces genuine mechanistic understanding. There is no abstraction to hide behind. Every concept from the theory (weighted sum, threshold, convergence) has a direct, tangible counterpart on the breadboard (which you could also see in the user interface (UI) file within this repo).
>
>This project emerged from a sketch I showed my professor: a biological neuron I had drawn by hand, and the words "this is what I want to build." That vision evolved into a perceptron, and after many late nights and numerous hours of fine-tuning, re-understanding concepts, and crashing out (every once in a while), here we are. Thank you, Paul, for guiding me in the right direction and for being there to troubleshoot. It's moments like these that remind me of the value of an education!
>>
>><img width="455" height="251" alt="IMG 5848" src="https://github.com/user-attachments/assets/f0c49719-b95d-49a7-9f5c-0e367ecbbefb" />
>>
>> *Figure 1: The 'Pitches and Sketches' Diagram that I showed Paul*
>

### How a Perceptron Relates to a Biological Neuron
>
>The structure of a biological neuron directly inspires the perceptron.[^1] In the brain, a neuron receives signals through its dendrites, integrates them in the cell body (soma), and—if the total signal crosses a threshold—fires an electrical spike down the axon, across the synaptic cleft, to the dendrites of the next neuron.
>
>>| Biological Neuron                          | Analog Perceptron                          |
>>| ------------------------------------------ | ------------------------------------------ |
>>| Dendrites receiving signals                | DIP switch inputs $(x_1, x_2)$             |
>>| Synapse strength                           | Potentiometer dial position $(w_1, w_2)$   |
>>| Axon hillock — summation + threshold check | Quad op-amp (summing node) + comparator    |
>>| Action potential (fires or doesn't)        | LED lights (green = fires, blue = doesn't) |
>>| Axon → next neuron                         | Output wire → next circuit stage           |
>>| Synaptic plasticity (learning)             | Turning the potentiometer dials by hand    |
>>
>>*Figure 2: A table showing the comparison between a biological neuron and an analog perceptron*
>
>When multiple perceptrons are combined, with the output of one feeding the inputs of others, you get a multi-layer neural network, capable of solving problems no single perceptron can. This is the architecture shown in the upper-left part of my notebook sketch (the multi-layer diagram with attention mechanisms), and the direction I intended for this project. 
>
>><img width="360" height="364" alt="Image from iLoveIMG" src="https://github.com/user-attachments/assets/263080ed-3014-4e39-acbf-7350bb65891e" />
>>
>>*Figure 3: My perceptron prototype theory on paper*
>
>My first iterations of this were on a cardboard box (as my base) with physical components I planned to use later (perceptron, switches, etc.). [Ridhwanlai's Works-like Prototype](https://drive.google.com/file/d/1gP8iWLKEKyuuV7yGvbxWcRd90X-XedA8/view)
>
>><img width="387" height="354" alt="BIG IMAGE" src="https://github.com/user-attachments/assets/a9f11f79-e6c6-4353-b978-5eafb5f90d64" />
>>
>>*Figure 4: My actual work-like perceptron paper prototype*
>
>><img width="540" height="327" alt="piazza posttt" src="https://github.com/user-attachments/assets/3e4f2ad8-7667-46a9-b0fa-e3b9b52f1406" />
>>
>> *Figure 5: Image of my Piazza post sharing the math behind the first layer of my neural network*
>
> Now for the part you've been waiting for (drumroll, please)... Before that, though, thank you for following along so far. If some of this makes absolutely no sense, great! I was in your shoes a few months ago. :)
>

## Perceptron Breakdown

### What Is a Perceptron?
>
> A perceptron is the simplest possible form of an artificial neural network. The main idea was conceived by Frank Rosenblatt in 1957; it's the fundamental building block from which every modern deep learning model—GPT, Claude, AlphaFold—ultimately descends.[^1]
>
> **At its core, a perceptron does three things:**
> * Takes multiple inputs and assigns each a numerical weight
> * Computes a weighted sum of those inputs plus a constant bias term
> * Applies a threshold (activation) function — outputs 1 if the sum exceeds the threshold, <0 if it does not
>   
> Mathematically, the output is: $\overline{y} = \sum_{i=1}^{n}(w_1x_1 + w_2x_2 + b)$
>
> **Where:**
> * $(x_1, x_2)$ are the binary inputs (switches: ON = 1, OFF = 0)
> * $(w_1, w_2)$ are the weights (set by the potentiometers)
> * $b$ is the bias (a constant offset that shifts the decision boundary)
> * $\sum_{i=1}^{n} ()$ is the activation function - positive sum → Class A, a zero, or negative sum → Class B
> 
> The perceptron is a binary linear classifier. Geometrically, the equation $(w_1x_1 + w_2x_2 + b) = 0$ defines a straight line in 2D space. Everything on one side of that line is Class A; everything on the other is Class B. Training the perceptron means adjusting $w_1, w_2,$ and $b$ until that line correctly separates all your training examples.[^2]
>
>><img width="422" height="295" alt="IMG 7031" src="https://github.com/user-attachments/assets/a6b6bc48-4719-4e00-bc62-1c03ac21fa3b" />
>>
>>*Figure 6: My perceptron diagram illustrating weighted inputs, summation, and activation function*
>

### Circuit Architecture — How My Physical Build Works
> 
> My circuit implements the perceptron equation $\overline{y} = \sum_{i=1}^{n}(w_1x_1 + w_2x_2 + b)$ in five distinct physical stages:
> 
> #### Stage 1: Inputs $(x_1, x_2)$
> Two positions of a Dual In-line Package (DIP) switch act as binary inputs. When a switch is ON, it connects 9V to the corresponding weight potentiometer, representing input = 1. When OFF, a pull-down resistor holds the line firmly at 0V (representing input = <0). The pull-down is essential: without it, an open switch leaves the input floating at an undefined voltage, producing garbage readings.
> 
>><img width="504" height="290" alt="INPUT" src="https://github.com/user-attachments/assets/b816b2dc-3579-4b48-a5a7-32207f8670ce" />
>>
>>*Figure 7: Inputs (dual in-line package) on the breadboard*
>
> #### Stage 2: Weights $(w_1, w_2, b)$
> Three 3386MP trimmer potentiometers act as the learnable weights. Each pot is wired as a voltage divider:
> * Left pin → input signal (from switch, or 9V directly for bias)
> * Right pin → GND
> * Center pin (wiper) → outputs a voltage proportional to how far the knob is turned
> 
> Turning the knob clockwise increases the weight (more voltage passes through), and counterclockwise approaches zero. The third pot is always connected to 9V regardless of switch state; this is the bias term $b$, which shifts the decision boundary away from the origin.
>
>><img width="544" height="329" alt="POTS" src="https://github.com/user-attachments/assets/51f4c01e-d0bb-41c3-8f6b-0ff73a2b245f" />
>>
>>*Figure 8: Weights (potentiometers) on the breadboard*
>
> #### Stage 3: Summation Node $\sum$
>The center (wiper) pins of all three pots connect through equal 220Ω series resistors into the inverting row input of one TL074CN quad op-amp (on the breadboard), which is wired as an inverting summing amplifier. With equal input resistors $(R)$ and a feedback resistor $(R_f)$, the output is: $V_{sum} = -\frac{R_f}{R}(w_1x_1 + w_2x_2 + b)$
>
>The output is inverted, which is accounted for in the comparator stage. Unlike passive resistive summing (where voltages interact and load each other), the op-amp actively buffers the result, giving a clean, accurate weighted $V_{sum}$ regardless of what the LEDs or downstream circuitry are doing. This is the key advantage of using the TL074 here rather than relying on passive summing alone.[^5]
>
>><img width="504" height="273" alt="SUMMM" src="https://github.com/user-attachments/assets/eb7c71be-2a5c-4189-a583-492c19a332c6" />
>>
>>*Figure 9: Summation row with resistors on the breadboard*
>
> There are two possible ways to get a reading of this sum. The first is by using a multimeter as the output reader. This is what I initially planned to do. Set it to DC Volts. Red probe on the summing row, black probe on GND. The voltage you read is $\overline{y}$. Above 4.5V = positive prediction. Below 4.5V = either a zero or a negative prediction. The second—and what I ended up building—is a Processing UI that reads the Arduino's analog measurement of this output voltage and visualizes the summation in real time as you turn the knobs. [See Processing Visualization](#processing-visualization)
>
> #### Stage 4: Activation Function (LM311 Comparator)
> The summed output from the TL074 feeds into the LM311 voltage comparator. The LM311 is a dedicated comparator IC — unlike a general-purpose op-amp, it is designed specifically for this job: it switches its output cleanly and quickly between HIGH and LOW based on which of its two inputs is larger. [^4]
>
>The LM311 compares two voltages:
> * V+ (non-inverting input) → $V_{sum}$ from the TL074 output
> * V− (inverting input) → $V_{ref}$ = 4.5V (set by two equal 220Ω resistors forming a voltage divider from 9V to GND)
> 
> When $V_{sum} > V_{ref}$: output goes HIGH → green LED lights (Class A — positive prediction)
>>
>><img width="458" height="283" alt="LED ON" src="https://github.com/user-attachments/assets/e76d3d95-46c5-48de-80bd-465c0ddff2c8" />
>> 
>> *Figure 10: My perceptron reading a HIGH output*
>
> When $V_{sum} < V_{ref}$: output goes LOW → blue LED lights (Class B — negative prediction)
>> 
>><img width="456" height="281" alt="LED OFF" src="https://github.com/user-attachments/assets/eed9935a-06aa-4e03-857b-c6e81074fb5e" />
>> 
>> *Figure 11: My perceptron reading a LOW output*
>
>><img width="326" height="288" alt="Comparator" src="https://github.com/user-attachments/assets/620b7981-2251-4dcc-87ab-538296a77551" /> <img width="326" height="288" alt="Quad Amp" src="https://github.com/user-attachments/assets/f356d67a-ae2e-4e36-b80b-3da94e92b48f" />
>>
>>*Figure 12: The LM311 comparator on the breadboard*
>>
>>*Figure 13: The TL074CN quad op-amp on the breadboard*
>
> #### Stage 5: Output $\overline{y}$
> Each LM311 output state drives one LED through a 220Ω current-limiting resistor (without this, the LED draws far too much current and burns out immediately — approximately $({9V − 2V}/{220Ω}) ≈ 32mA$, safely within the LED's rated range).

### Processing Visualization
>
> As mentioned above, to make the circuit's internal behaviour visible and legible, a companion Processing sketch that communicates with an Arduino over serial was developed.
>> The UI in Processing was generated by Claude, an LLM created by Anthropic.
> 
> **The Arduino reads:**
> * The voltage at the summing node (Row S) via an analog input pin
> * The state of the two DIP switch inputs
> * It then streams this data as CSV over serial $(x_1, x_2, V_{sum})$ to a Processing sketch running on a connected laptop
>   
> **The Processing visualization displays:**
> * A real-time dial/gauge showing the current value of $V_{sum}$ — styled with a CRT aesthetic, sweeping left to right as you turn the potentiometer knobs
> * A 2D decision boundary plot showing the current classification line $(w_1x_1 + w_2x_2 + b) = 0$, updating live as the pots are turned
> * The current values of $(w_1x_1 + w_2x_2)$ and $b$ displayed numerically as the knobs are adjusted
> * LED indicators mirroring the physical LEDs, so the audience can see the binary output clearly on-screen
>   
>> ***Voltage safety note** if you're interested in replicating: The Arduino's analog inputs operate at 5V maximum. The op-amp summing output (0–9V) was stepped down via a 2:1 voltage divider (two equal resistors) before entering the analog pin. This prevents damage to the Arduino.*
>
>> https://github.com/user-attachments/assets/672e73bd-2619-43fe-bad7-60b572923499
>> 
>>*Video 1: An explanation of how the Perceptron is working, turning on switches and weights*
>

### Jupyter Notebook — Software Perceptron
>
>> The file is within my GitHub repo.
>
> To deepen my understanding of the perceptron learning algorithm and go beyond what a physical single-layer analog circuit can do, a software implementation was built in [JupyterLab](https://colab.research.google.com/drive/1Jk_zGaJPHm822mu-ubwsRWYv3TLRblkk?usp=sharing), following along a YouTube demonstration by ethicalPap's Perceptron Algorithm Learning series.[^3]
> 
> **The notebook implements:**
> * The perceptron learning rule from scratch in Python (no ML libraries)
> * Visualization of the decision boundary updating over training epochs
> * Training on the AND, and OR logic gate problems
> * A demonstration of why XOR fails (not linearly separable) — the motivation for multi-layer networks

### Limitations — Why XOR is Impossible (for my Perceptron)
>
> The fundamental limitation of a single-layer perceptron is that it can only solve linearly separable problems; problems where a single straight line can divide Class A from Class B. The XOR problem is the canonical example of a problem that is not linearly separable:
>
>>| $x_1$| $x_2$| XOR output |
>>| ---- | ---- | ---------- |
>>| 0    | 0    | 0          |
>>| 0    | 1    | 1          |
>>| 1    | 0    | 1          |
>>| 1    | 1    | 0          |
>>
>> *Figure 14: A table showing the output of an XOR* 
>
> No single straight line can separate the 1 outputs from the 0 outputs, as they form an alternating checkerboard pattern. This is why XOR was the problem that broke the first perceptron hype cycle in the 1960s.[^6]
>
> The solution is multiple layers (the multi-layer perceptron (MLP) shown in the upper-right of my original notebook sketch; figure 3). With a hidden layer, the network can learn non-linear boundaries by combining multiple linear decisions. This is the architecture behind every modern deep learning model, and the direction this project is intended to grow. My goal is to ensure my 'hidden layer' remains interpretable, and understanding the concepts in this project has gone a long way toward achieving that.
>

## Conclusion

### Challenges
>
> Before committing to the physical build, three full circuit simulations were developed in TinkerCad Circuits to validate the design and catch wiring errors before they became real problems. Tinkercad runs simulations, probe voltages, and flip switches in a virtual environment, which was invaluable for debugging a circuit this interconnected. It also made it much easier to work through the challenges I ran into because I could try a few different versions to get to the core problem.
>
> 
> **Prototype 1:** Basic summing network (op-amp working). Validated that my three potentiometers wired to a shared row through equal resistors were in fact producing a voltage proportional to the weighted sum of the inputs. I struggled with summation here, and my nodes were not directly feeding into the LED.
>
>><img width="516" height="191" alt="Version 1" src="https://github.com/user-attachments/assets/97497637-3591-4074-b67f-c8933e86ea4e" />
>>
>>*Figure 15: Prototype 1 in TinkerCad*
>
> **Prototype 2**: Single comparator with fixed threshold, including having a fully operational op-amp comparator stage with a fixed 4.5V V_{ref} divider. Was able to confirm mutually exclusive LED behaviour, exactly one LED on at a time, cleanly switching as the summing node crossed the threshold. The problem was that my summation was subtracting from the row, never crossing the threshold.
>
>><img width="516" height="245" alt="Version 2" src="https://github.com/user-attachments/assets/ed90c3e4-3ded-4b1d-907d-e003ea585121" />
>>
>>*Figure 16: Prototype 2 in TinkerCad*
>
> **Prototype 3:** Full circuit with dual comparators and DIP switch inputs. Complete design: two-position DIP switch inputs with pull-down resistors, three pots (two for weights, one for bias), resistive summing network, dual TL074 comparator outputs, and both LEDs with current-limiting resistors. This is the final design that made me feel comfortable enough to transfer what I knew onto a physical breadboard.
>
>><img width="516" height="190" alt="Version 3" src="https://github.com/user-attachments/assets/ad3b7411-2258-4fe7-a3ff-86b5ab8297b4" />
>>
>>*Figure 17: Prototype 3 in TinkerCad*
>

### Reflections
>
>Safe to say, this went beyond what I expected of myself (for a personal project). From numerous reworkings of the system layout and setup, to debugging summation principles with resistors, to rebuilding the entire approach multiple times (including multiple dead ends in TinkerCad and one complete pivot away from a standalone neuron design), this project took almost everything. And I still gave.
>
>What came out the other side is something I'm genuinely proud to produce: a (mostly) working, physical, trainable machine learning model that you can hold, probe with a multimeter, train by turning dials, and watch make decisions in real time. The next step is to take the understanding built here and create systems that do really cool things (essentially run with this). Multi-layer networks. Real-time biosignal classification. Transparent, interpretable models for healthcare contexts where the stakes are high enough that "I don't know why it decided that" is not an acceptable answer.
>
>These may seem like big dreams, but there was a point in the semester when I didn’t think I’d make it. But with a bit of twisting and turning, I was able to reach the desired output. Thank you!
>
> #### Moments Along The Way
>>
>><img width="336" height="275" alt="IMG 6820" src="https://github.com/user-attachments/assets/5a504a4d-bfe5-46c5-b09e-159553e3afa6" />
>>
>>*Figure 18: Late-night working session in my room (taken in early March)*
>
>><img width="476" height="239" alt="IMG 6834" src="https://github.com/user-attachments/assets/6793beea-87e5-4861-ba0c-b8a4469cdb16" />
>>
>>*Figure 19: At the library! Shoutout to Subway for fueling me throughout this journey*
>
>>https://github.com/user-attachments/assets/37a721df-7193-4a90-aedf-ae5dd85f1f77
>>
>>*Video 2: Time-lapse of me pulling an all-nighter ahead of my presentation day (it was a success!)*
>

### References & Citations
>
> [^1]: Rosenblatt, F. (1958). *The Perceptron: A probabilistic model for information storage and organization in the brain.* Psychological Review, 5(), 38–408
> [^2]: [GeeksForGeeks — What Is a Perceptron?](https://www.geeksforgeeks.org/deep-learning/what-is-perceptron-the-simplest-artificial-neural-network/)
> [^3]: [ethicalPap — Machine Learning 101, Lecture 2: Perceptron Algorithm](https://www.youtube.com/watch?v=ziDrxd_JTvs)
> [^4]: [Texas Instruments — LM311 Datasheet](https://www.ti.com/product/LM311?ds_k=LM311+Datasheet&DCM=yes&gad_campaignid=14388345080&gbraid=0AAAAAC08F2oICd-7U2a7-dLEgwnclq0)
> [^5]: [Texas Instruments — TL074 Datasheet](https://www.ti.com/product/TL074)
> [^6]: Minsky, M. & Papert, S. (199). *Perceptrons.* MIT Press
>
