# Steganography Tools

---

To implement`Tensor steganography`, we need to develop two Python functions:`encode_lsb`to embed data within a tensor's`least significant bits`(`LSBs`), and`decode_lsb`to reverse the process, and retrieve it. These two functions rely on the`struct`module for conversions between floating-point numbers and their raw byte representations, which is essential for bit-level manipulation.

Code:python```
`importstruct`
```

## Encoding Logic

The`encode_lsb`function embeds a byte string (`data_bytes`) into the LSBs of a`float32`tensor (`tensor_orig`), using a specified number of bits (`num_lsb`) per tensor element.

We start by defining the function and performing initial validations. These checks ensure the input tensor`tensor_orig`is of the`torch.float32`data type, as our LSB manipulation technique is specific to this format. We also need to confirm that`num_lsb`is within an acceptable range (1 to 8 bits). To prevent modification of the original input, we work only on a`clone`of the tensor.

Code:python```
`defencode_lsb(tensor_orig:torch.Tensor,data_bytes:bytes,num_lsb:int)->torch.Tensor:"""Encodes byte data into the LSBs of a float32 tensor (prepends length).

    Args:
        tensor_orig: The original float32 tensor.
        data_bytes: The byte string to encode.
        num_lsb: The number of least significant bits (1-8) to use per float.

    Returns:
        A new tensor with the data embedded in its LSBs.

    Raises:
        TypeError: If tensor_orig is not a float32 tensor.
        ValueError: If num_lsb is not between 1 and 8.
        ValueError: If the tensor does not have enough capacity for the data.
    """iftensor_orig.dtype!=torch.float32:raiseTypeError("Tensor must be float32.")ifnot1<=num_lsb<=8:raiseValueError("num_lsb must be 1-8. More bits increase distortion.")tensor=tensor_orig.clone().detach()# Work on a copy`
```

Next, we prepare the data for embedding. The tensor is flattened to simplify element-wise iteration. Here, the length of`data_bytes`is determined and then packed as a 4-byte, big-endian unsigned integer using`struct.pack(">I", data_len)`. This length prefix is prepended to`data_bytes`to form`data_to_embed`. This step ensures the decoder can ascertain the exact size of the hidden payload.

Code:python```
`n_elements=tensor.numel()tensor_flat=tensor.flatten()# Flatten for easier iterationdata_len=len(data_bytes)# Prepend the length of the data as a 4-byte unsigned integer (big-endian)data_to_embed=struct.pack(">I",data_len)+data_bytes`
```

A capacity check is then performed. We calculate the`total_bits_needed`for`data_to_embed`(length prefix + payload) and compare this to the tensor's`capacity_bits`(derived from`n_elements * num_lsb`). If the tensor lacks sufficient capacity, a`ValueError`is raised, as attempting to embed the data would fail. This ensures we don't try to write past the available space.

Code:python```
`total_bits_needed=len(data_to_embed)*8capacity_bits=n_elements*num_lsbiftotal_bits_needed>capacity_bits:raiseValueError(f"Tensor too small: needs{total_bits_needed}bits, but capacity is{capacity_bits}bits. "f"Required elements:{(total_bits_needed+num_lsb-1)//num_lsb}, available:{n_elements}.")`
```

We then initialize variables to manage the bit-by-bit embedding loop:`data_iter`allows iteration over`data_to_embed`,`current_byte`holds the byte being processed, and`bit_index_in_byte`tracks the current bit within that byte (from 7 down to 0),`element_index`points to the current tensor element, and`bits_embedded`counts the total bits successfully stored.

Code:python```
`data_iter=iter(data_to_embed)# To get bytes one by onecurrent_byte=next(data_iter,None)# Load the first bytebit_index_in_byte=7# Start from the MSB of the current_byteelement_index=0# Index for tensor_flatbits_embedded=0# Counter for total bits embedded`
```

The main embedding occurs in a`while`loop, processing one tensor element at a time. For each`float32`value, its 32-bit integer representation is obtained using`struct.pack`and`struct.unpack`. A`mask`is created to target the`num_lsb`LSBs, and an inner loop then extracts`num_lsb`bits from`data_to_embed`(via`current_byte`and`bit_index_in_byte`), assembling them into`data_bits_for_float`. This process continues until all payload bits are gathered for the current float or the payload ends.

Code:python```
`whilebits_embedded<total_bits_neededandelement_index<n_elements:ifcurrent_byteisNone:# Should not happen if capacity check is correctbreakoriginal_float=tensor_flat[element_index].item()# Convert float to its 32-bit integer representationpacked_float=struct.pack(">f",original_float)int_representation=struct.unpack(">I",packed_float)[0]# Create a mask for the LSBs we want to modifymask=(1<<num_lsb)-1data_bits_for_float=0# Accumulator for bits to embed in this floatforiinrange(num_lsb):# For each LSB position in this floatifcurrent_byteisNone:# No more data bytesbreakdata_bit=(current_byte>>bit_index_in_byte)&1data_bits_for_float|=data_bit<<(num_lsb-1-i)bit_index_in_byte-=1ifbit_index_in_byte<0:# Current byte fully processedcurrent_byte=next(data_iter,None)# Get next bytebit_index_in_byte=7# Reset bit indexbits_embedded+=1ifbits_embedded>=total_bits_needed:# All data embeddedbreak`
```

With`data_bits_for_float`prepared, we embed these bits into the tensor element. First, the LSBs of the`int_representation`are cleared using a bitwise`AND`with the inverted`mask`. Then,`data_bits_for_float`are merged into these cleared positions using a bitwise`OR`. The resulting`new_int_representation`is converted back to a`float32`value using`struct.pack`and`struct.unpack`. This new float, containing the embedded data bits, replaces the original value in`tensor_flat`. The`element_index`is then incremented.

Code:python```
`# Clear the LSBs of the original float's integer representationcleared_int=int_representation&(~mask)# Combine the cleared integer with the data bitsnew_int_representation=cleared_int|data_bits_for_float# Convert the new integer representation back to a floatnew_packed_float=struct.pack(">I",new_int_representation)new_float=struct.unpack(">f",new_packed_float)[0]tensor_flat[element_index]=new_float# Update the tensorelement_index+=1`
```

After the loop finishes, a confirmation message is printed detailing the number of bits encoded and tensor elements used. The modified`tensor`(which reflects changes made to its flattened view,`tensor_flat`) is then returned.

Code:python```
`print(f"Encoded{bits_embedded}bits into{element_index}elements using{num_lsb}LSB(s) per element.")returntensor`
```

## Decoding Logic

The`decode_lsb`function reverses the encoding, extracting hidden data from a`tensor_modified`. It requires the tensor and the same`num_lsb`value used during encoding.

Initial setup validates the tensor type (`float32`) and`num_lsb`range. The tensor is flattened, and a`shared_state`dictionary is used to manage`element_index`across calls to a nested helper function, ensuring that bit extraction resumes from the correct position in the tensor.

Code:python```
`defdecode_lsb(tensor_modified:torch.Tensor,num_lsb:int)->bytes:"""Decodes byte data hidden in the LSBs of a float32 tensor.
    Assumes data was encoded with encode_lsb (length prepended).

    Args:
        tensor_modified: The float32 tensor containing the hidden data.
        num_lsb: The number of LSBs (1-8) used per float during encoding.

    Returns:
        The decoded byte string.

    Raises:
        TypeError: If tensor_modified is not a float32 tensor.
        ValueError: If num_lsb is not between 1 and 8.
        ValueError: If tensor ends prematurely during decoding or length/payload mismatch.
    """iftensor_modified.dtype!=torch.float32:raiseTypeError("Tensor must be float32.")ifnot1<=num_lsb<=8:raiseValueError("num_lsb must be 1-8.")tensor_flat=tensor_modified.flatten()n_elements=tensor_flat.numel()shared_state={'element_index':0}`
```

The nested`get_bits(count)`function is responsible for extracting a specified`count`of bits from the tensor's LSBs. It iterates through`tensor_flat`elements, starting from`shared_state['element_index']`. For each float, it obtains its integer representation, masks out the`num_lsb`LSBs, and appends these bits to a list until`count`bits are collected, and`shared_state['element_index']`is updated after each element. If the tensor ends before`count`bits are retrieved, a`ValueError`is raised.

Code:python```
`defget_bits(count:int)->list[int]:nonlocalshared_state 
        bits=[]whilelen(bits)<countandshared_state['element_index']<n_elements:current_float=tensor_flat[shared_state['element_index']].item()packed_float=struct.pack(">f",current_float)int_representation=struct.unpack(">I",packed_float)[0]mask=(1<<num_lsb)-1lsb_data=int_representation&maskforiinrange(num_lsb):bit=(lsb_data>>(num_lsb-1-i))&1bits.append(bit)iflen(bits)==count:breakshared_state['element_index']+=1iflen(bits)<count:raiseValueError(f"Tensor ended prematurely. Requested{count}bits, got{len(bits)}. "f"Processed{shared_state['element_index']}elements.")returnbits`
```

Decoding begins by calling`get_bits(32)`to retrieve the 32-bit length prefix. These bits are then converted into an integer,`payload_len_bytes`, representing the length of the hidden payload in bytes. Appropriate error handling is included for this critical step.

Code:python```
`try:length_bits=get_bits(32)# Decode the 32-bit length prefixexceptValueErrorase:raiseValueError(f"Failed to decode payload length:{e}")payload_len_bytes=0forbitinlength_bits:payload_len_bytes=(payload_len_bytes<<1)|bit`
```

If`payload_len_bytes`is zero, it indicates no payload is present, and an empty byte string is returned. Otherwise,`get_bits`is called again to retrieve`payload_len_bytes * 8`bits, which constitute the actual payload. The`get_bits`function seamlessly continues from where it left off, thanks to the persisted`shared_state['element_index']`.

Code:python```
`ifpayload_len_bytes==0:print(f"Decoded length is 0. Returning empty bytes. Processed{shared_state['element_index']}elements for length.")returnb""# No payload if length is zerotry:payload_bits=get_bits(payload_len_bytes*8)# Decode the actual payloadexceptValueErrorase:raiseValueError(f"Failed to decode payload (length:{payload_len_bytes}bytes):{e}")`
```

The extracted`payload_bits`are then reconstructed into bytes. We iterate through`payload_bits`, accumulating them into`current_byte_val`. When 8 bits are collected (tracked by`bit_count`), the complete byte is appended to`decoded_bytes`(a`bytearray`), and the accumulators are reset.

Code:python```
`decoded_bytes=bytearray()current_byte_val=0bit_count=0forbitinpayload_bits:current_byte_val=(current_byte_val<<1)|bit
        bit_count+=1ifbit_count==8:# A full byte has been assembleddecoded_bytes.append(current_byte_val)current_byte_val=0# Reset for the next bytebit_count=0# Reset bit counter`
```

Finally, the`decoded_bytes`bytearray is converted to an immutable`bytes`object and returned, completing the data extraction.

Code:python```
`print(f"Decoded{len(decoded_bytes)}bytes. Used{shared_state['element_index']}tensor elements with{num_lsb}LSB(s) per element.")returnbytes(decoded_bytes)`
```