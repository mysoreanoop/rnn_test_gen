import os

# run simulation
os.chdir('/home/anoop/uw/mlfuzz/black-parrot-sim/rtl/bp_top/syn/')
os.system('make build.sc sim.sc COSIM_P=0 CMT_TRACE_P=0 SUITE=riscv-tests PROG=rv64ui-p-add')

# mutate program
os.system('vi /home/anoop/')

# compile mutated program
os.chdir('/home/anoop/uw/black-parrot-sim/sdk/riscv-tests/isa/rv64ui')
os.system('../../../install/bin/riscv64-unknown-elf-dramfs-gcc add.S -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles -DBAREMETAL -I../../../install/include -I../../env/p -I../macros/scalar -T../../env/p/link.ld ../../../perch/atomics.S ../../../perch/muldiv.S ../../../perch/exception.S ../../../perch/emulation.c ../../../perch/bp_utils.c -o /home/anoop/uw/596/text_gen/data/test.riscv')
