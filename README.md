## MPI Pancake 
MPI_Pancake is a lightweight, LD_PRELOADable library which tries to optimize MPI GPU communications from MPI_Derived types by hooking MPI_Isend and MPI_Irecv calls and then performing GPU packing and unpacking. The library is developed for Vlasiator but can be used with all kind of software that do MPI comms with GPU data (especially those using MPI Derived types).

### Limitations
**Curretnly MPI_Pancake will flatten `STRUCT HINDEXED MPI_BYTE` and `STRUCT MPI_BYTE` nested types that are using GPU memory but this is easily expandable to more 
complex or simpler types.** If you are interested in using MPI_Pancake with your data please open an issue.

**MPI_Pancake currently hooks Isend/Irecv. This can also be easily expanded to non blocking/collective calls but I would prefer to do this when the need arises instead
of trying to catch everything at once.**

## Buiding 
You can use either the Makefile provided and use either `make USE_CUDA=1` or `make USE_HIP=1`. However sometimes the MPI wrappers are weird so you might want to use the following one liners if the Makefile fails (one day I might add cmake support). This will build `libmpipancake.so`,`libmpisniffer.so` and `test`.

### HIP systems
`hipcc -O3 -std=c++17 -Wno-unused-result -fPIC -shared -x hip mpi_pancake.cpp  -o libmpipancake.so`
`hipcc -O3 -std=c++17 -Wno-unused-result -fPIC -shared -x hip mpi_sniffer.cpp  -o libmpisniffer.so`

### CUDA systems
`nvcc -O3 -std=c++17 -ccbin mpicxx -Xcompiler=" -fPIC -shared -Wall " -x cu mpi_pancake.cpp -o libmpipancake.so -lmpi`
`nvcc -O3 -std=c++17 -ccbin mpicxx -Xcompiler=" -fPIC -shared -Wall " -x cu mpi_sniffer.cpp -o libmpisniffer.so -lmpi`


## Usage
### libmpipancake.so:  
Just add `LD_PRELOAD=./libmpipancake.so` in front of your mpirun/srun command pretty much.
### libmpisniffer.so:  
Just add `LD_PRELOAD=./libmpisniffer.so` in front of your mpirun/srun command.
### test:
Run `LD_PRELOAD=./libmpipancake.so mpirun/srun <whatever flags you need> -n 2 test`.


### Expected output for test:
```
   LD_PRELOAD=./libmpipancake.so mpirun -n 2 ./test
      ========= MPI_PANCAKE Initialized =========
      ========= MPI_PANCAKE Initialized =========
      Bytes           | Pancake (ms) | Control (ms) | SANE
      -----------------------------------------------------------
      3413            | 6.0052       | 0.0163       | OK
      6826            | 1.9103       | 0.0202       | OK
      13653           | 1.5135       | 0.0323       | OK
      27306           | 1.5760       | 0.0283       | OK
      54613           | 1.7283       | 0.0301       | OK
      109226          | 1.8979       | 0.0392       | OK
      218453          | 2.1469       | 0.0647       | OK
      436906          | 2.2533       | 0.1038       | OK
      873813          | 2.6399       | 0.1826       | OK
      1747626         | 3.4685       | 0.3523       | OK
      ========= MPI_PANCAKE Finalized =========
      ========= MPI_PANCAKE Finalized =========

```
You can tell MPI_Pancake is working by the Initialized/Finalized messages.

### Expected output for libmpisniffer:
You should pretty much get a description of the messages you are sending upong MPI_Finalize. The following is from a Vlasiator test.
Nested types are also supported like `STRUCT HINDEXED MPI_BYTE`.
```
    --- MPI Sniffer ---
      [MPI_Send   CALLS]
        MPI_INT                            -> 158088     calls, 617.53 KB
        ---------------------------------- Total: 617.53 KB
      [MPI_Isend  CALLS]
        STRUCT MPI_BYTE                    -> 32824004   calls, 0 B
        MPI_BYTE                           -> 385568     calls, 3.37 GB
        STRUCT HINDEXED MPI_BYTE           -> 2209231    calls, 793.11 GB
        SUBARRAY CONTIGUOUS MPI_BYTE       -> 1517952    calls, 374.29 GB
        STRUCT MIXED                       -> 2226741    calls, 192.43 GB
        ---------------------------------- Total: 1.33 TB
      [MPI_Recv   CALLS]
        (None)
      [MPI_Irecv  CALLS]
        MPI_INT                            -> 158088     calls, 617.53 KB
        MPI_BYTE                           -> 1320332    calls, 4.90 GB
        STRUCT MPI_BYTE                    -> 32824004   calls, 0 B
        STRUCT HINDEXED MPI_BYTE           -> 2209231    calls, 793.11 GB
        SUBARRAY CONTIGUOUS MPI_BYTE       -> 1517952    calls, 374.29 GB
        STRUCT MIXED                       -> 2226741    calls, 192.43 GB
        ---------------------------------- Total: 1.33 TB
```
