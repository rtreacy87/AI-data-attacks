# Pickles and Tensor Steganography

---

The proliferation of pre-trained models, readily available from repositories like`Hugging Face`or`TensorFlow Hub`, offers immense convenience but also[present a significant attack surface](https://arxiv.org/abs/2107.08590). An attacker could modify a benign pre-trained model to embed hidden data or even malicious code directly within the model's parameters.

An important note: The methodologies explored within this section are very real attack vectors, but the actual implementation is more hypothetical and for demonstration purposes, than a guide on how to embed and distribute malware.

`The primary vector for this type of attack often lies not within the sophisticated mathematics of the neural network itself, but in the fundamental way models are saved and loaded.`

`pickle`is Python's standard way to`serialize`an object (convert it into a byte stream) and`deserialize`it (reconstruct the object from the byte stream). While powerful, deserializing data from an untrusted source with`pickle`is inherently dangerous. This is because`pickle`allows objects to define a special method:`__reduce__`. When`pickle.load()`encounters an object with this method, it calls`__reduce__`to get instructions on how to rebuild the object, and these instructions typically involve a callable (like a class constructor or a function) and its arguments.

An adversary can exploit this by creating a custom class where`__reduce__`returns a dangerous callable, such as the built-in`exec`function or`os.system`, along with malicious arguments (like a string of code to execute or a system command). When`pickle.load()`deserializes an instance of this malicious class, it blindly follows the instructions returned by`__reduce__`, leading to arbitrary code execution on the machine. The official Python documentation even explicitly warns:**"Warning: The pickle module is not secure. Only unpickle data you trust."**

PyTorch's`torch.save(obj, filepath)`uses`pickle`to save model instances.`torch.load(filepath)`uses`pickle.load()`internally to deserialize the object(s) from the file. This means`torch.load`inherits the security risks of`pickle`.

Recognizing this significant risk, PyTorch introduced the`weights_only=True`argument for`torch.load`. When set (in newer versions its default state is true),`torch.load(filepath, weights_only=True)`drastically restricts what can be loaded. It uses a safer unpickler that only allows basic Python types essential for loading model parameters (tensors, dictionaries, lists, tuples, strings, numbers, None) and refuses to load arbitrary classes or execute code via`__reduce__`.

This attack targets the specific vulnerability exposed when`torch.load(filepath)`is called explicitly using`weights_only=False`. In this insecure mode,`torch.load`behaves like`pickle.load`and will execute malicious code embedded via`__reduce__`.

While unsafe deserialization provides the mechanism for execution, the model's internal structure - its vast collection of numerical parameters - provides a medium where malicious data, payloads, or configuration details can be hidden.

## Understanding Neural Network Parameters

As you know, neural networks learn by optimizing numerical parameters, primarily`weights`associated with connections and`biases`associated with neurons. These learned parameters represent the model's acquired knowledge and need to be stored efficiently for saving, sharing, or deployment.

The standard way to organize and store these large sets of`weights`and`biases`is using data structures called`tensors`. A`tensor`is fundamentally a multi-dimensional array, extending the concepts of`vectors`(`1D tensors`) and`matrices`(`2D tensors`) to accommodate data with potentially more dimensions. For example, the`weights`linking neurons between two fully connected layers might be stored as a 2D tensor (a matrix), whereas the filters learned by a convolutional layer are often represented using a 4D tensor.

The entire collection of all these learnable parameter`tensors`belonging to a model is what is referred to as its`state dictionary`(often abbreviated as`state_dict`in frameworks like PyTorch). When you save a trained model's parameters, you are typically saving this`state dictionary`. It is within the numerical values held in these`tensors`that techniques like`Tensor steganography`aim to hide data.

## Tensor Steganography

The practice of hiding information within the numerical parameters of a neural network model is known as`Tensor Steganography`. This technique leverages the fact that models contain millions, sometimes billions or even trillions, of parameters, typically represented as`floating-point numbers`.

The core idea is to alter the parameters in a way that is statistically inconspicuous and has minimal impact on the model's overall performance, thus avoiding detection. This hidden data might be the malicious payload itself, configuration for malware, or a trigger activated by the code executed via the`pickle`vulnerability.`Tensor steganography`, therefore, serves as a method to use the model's parameters as a data carrier, complementing other vulnerabilities like unsafe deserialization that provide the execution vector. A common approach to achieve this stealthy modification is to alter only the`least significant bits`(`LSBs`) of the floating-point numbers representing the parameters.

## The Structure of Floating-Point Numbers

To understand how LSB modification enables`Tensor steganography`, we need to look at how computers represent decimal numbers. The parameters in`tensors`are most commonly stored as`floating-point numbers`, typically conforming to the`IEEE 754`standard. The`float32`(single-precision) format is frequently used.

For a`float32`, each number is stored using 32 bits allocated to three distinct components according to a standard layout.



First, the`Sign Bit`(ss),
which is the`most significant bit`(`MSB`)
overall (Bit 31), determines if the number is positive (`0`)
or negative (`1`). Second, the`Exponent`(EstoredE_{stored}),
uses the next 8 bits (30 down to 23) to represent the number’s scale or
magnitude, stored with a`bias`(typically 127 for`float32`) to handle both large and small values. The actual
exponent isE=Estored−biasE = E_{stored} - \text{bias}.
Third, the`Mantissa`or`Significand`(mm)
uses the remaining 23 least significant bits (LSBs) (Bit 22 down to 0)
to represent the number’s precision or significant digits. The value is
typically calculated as:







Value=(−1)s×(1.m)×2(Estored−bias)\text{Value} = (-1)^s \times (1.m) \times 2^{(E_{stored} - \text{bias})}





Here,(1.m)(1.m)represents the implicit leading`1`combined with the
fractional part represented by the mantissa bitsmm.
Note that this formula applies to normalized numbers; special
representations exist for zero, infinity, and denormalized numbers, but
the core principle relevant to steganography lies in manipulating the
mantissa bits of typical weight values.





To make this clearer, let’s break down the float`0.15625`. The first step is to represent this decimal number
in binary. We can achieve this through a process of repeated
multiplication of the fractional part by 2. Starting with0.156250.15625,
multiplying by 2 gives0.31250.3125,
and we note the integer part is`0`. Taking the new
fractional part,0.3125×2=0.6250.3125 \times 2 = 0.625,
the integer part is again`0`. Continuing this process,0.625×2=1.250.625 \times 2 = 1.25,
yielding an integer part of`1`. We use the remaining
fractional part,0.25×2=0.50.25 \times 2 = 0.5,
which gives an integer part of`0`. The final step is0.5×2=1.00.5 \times 2 = 1.0,
with an integer part of`1`. By collecting the integer parts
obtained in sequence (`0`,`0`,`1`,`0`,`1`), we form the binary fraction0.0010120.00101_2.
Thus,0.15625100.15625_{10}is equivalent to0.0010120.00101_2.





Next, this binary number needs to be`normalized`for the
IEEE 754 standard.`Normalization`involves rewriting the
number in the form1.fractional_part2×2exponent1.\text{fractional\\_part}_2 \times 2^{\text{exponent}}.
To convert0.0010120.00101_2to this format, we shift the binary point three places to the right,
resulting in1.0121.01_2.
To preserve the original value after shifting right by three places, we
must multiply by2−32^{-3}.
This gives the normalized form1.012×2−31.01_2 \times 2^{-3}.





From this normalized representation, we can directly extract the
components required for the`float32`format. First, the Sign
bitssis`0`, as`0.15625`is a positive number. Second,
the actual exponent is identified asE=−3E = -3,
determined from the2−32^{-3}factor in the normalized form. The exponent stored in the`float32`format uses a bias (127), so the stored exponent isEstored=E+bias=−3+127=124E_{stored} = E + \text{bias} = -3 + 127 = 124.
In binary, this value is`01111100`. Finally, the mantissa
bitsmmare derived from the fractional part following the implicit leading`1`in the normalized form
(1.01_21.\underline{01}_2).
These bits start with`01`and are then padded with trailing
zeros to meet the 23-bit requirement for the mantissa field, giving`01000000000000000000000`.



The diagram below displays these exact bits (`0 01111100 010...0`) overlaid onto the corresponding fields, boxes separating out the individual bit positions.

![Diagram of IEEE 754 Float32 Bit Structure for 0.15625, showing Sign (1 bit), Exponent (8 bits), and Mantissa (23 bits) with bit positions labeled.](https://academy.hackthebox.com/storage/modules/302/ieee_754.png)

The part that we are interested in for steganography lies within the`mantissa`field (Bits 22 down to 0). The bits towards the left (starting with`0`at Bit 22 in the example) are the`most significant bits`(`MSBs`) of the mantissa, contributing more to the number's value. Conversely, the bits towards the far right (ending with`0`at Bit 0 in the example) are the`least significant bits`(`LSBs`) of the mantissa.

The previous diagram showed the structure for`0.15625`. Now, let's visually compare the effect of flipping different bits within its mantissa. The core idea of LSB steganography relies on the fact that changing the least significant bits has a minimal impact on the overall value, making the change hard to detect. Conversely, changing more significant bits causes a much larger, more obvious alteration.

The following two diagrams illustrate this. We start with our original value`0.15625`.

First, we flip only the LSB of the mantissa (Bit 0).

![Diagram of Float32 LSB Flip (Bit 0) showing original 0.15625 modified to 0.156250014901161, with Sign, Exponent, and Mantissa bit positions labeled.](https://academy.hackthebox.com/storage/modules/302/lsb_flip.png)



As you can see, flipping Bit 0 resulted in an extremely small change
to the overall value (approximately1.49×10−81.49 \times 10^{-8}).
This magnitude of change is often negligible in the context of deep
learning model weights, potentially falling within the model’s inherent
noise or tolerance levels.





Next, we flip the MSB of the mantissa (Bit 22, the leftmost bit within the mantissa field).

![Diagram of Float32 Mantissa MSB Flip (Bit 22) showing original 0.15625 modified to 0.21875, with Sign, Exponent, and Mantissa bit positions labeled.](https://academy.hackthebox.com/storage/modules/302/msb_flip.png)



Flipping Bit 22 caused a significant jump in the value (a change of0.06250.0625).
This change is orders of magnitude larger than flipping the LSB and
would likely alter the model’s behavior noticeably, making it a poor
choice for hiding data stealthily.





This comparison clearly demonstrates why LSBs are targeted in steganography. Altering them introduces minimal numerical error, preserving the approximate value and function of the number (like a weight or bias), thus hiding the embedded data effectively. Modifying more significant bits would likely corrupt the model's performance, revealing the tampering.