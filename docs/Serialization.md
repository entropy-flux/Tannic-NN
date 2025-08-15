## Serialization


### Header
- Magic Number
- Protocol Version
- Request Type
- Header Checksum
- Total Message Length

### Metadata

- Number of tensors 
    - rank 
    - shape
    - strides 
    - dtype
    - offset

### Data

- Raw tensors bytes







📦 Revised structure (minimal changes)

Header (fixed size)
Magic Number (4 bytes)
Protocol Version (1 byte)
Request Type (1 byte)
Header Checksum (2 bytes)
Total Message Length (8 bytes)

Metadata (variable) 
For each tensor:
Rank (1 byte)
Shape (rank × 8 bytes)
Strides (rank × 8 bytes)
Dtype code (1 byte)
Offset (8 bytes)
Data

Raw tensor bytes (possibly compressed)
Footer
Payload checksum/hash (4–32 bytes depending on method)