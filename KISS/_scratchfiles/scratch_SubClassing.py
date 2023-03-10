from AIMM_simulator import Sim, Cell

# Basic test to see it works
sim = Sim()
cell0 = Cell(sim=sim)
print(f'Cell0 is at position {cell0.xyz}')

# Now lets subclass the Cell class and call it NR_Cell
class NR_Cell(Cell):
    # What happens if we add an init?
    def __init__(self):
        print('This is a 5G NR Cell.')
    pass

# Create an NR_Cell object
nr_cell_0 = NR_Cell()

# Can we still access the xyz from the parent class?
# print(f'NR_Cell[1] position is {nr_cell_0.xyz}')
# NOPE! This is becuase we've overwritten the init of the Cell class in the NR_Cell class.

# Lets fix the NR_Cell class to inherit the Cell methods
class NR_Cell_v1(Cell):
    def __init__(self, nr_sim):
        # Here we'll use the super() method to pull in the init of the Cell class
        # The Cell class must have `sim` as the first argument that gets passed to it, so we have to put it in our init, to be able to pass it 'up' to the parent.
        super().__init__(sim=nr_sim)

# Can we access the position of a new NR_Cell_v1 object?
nr_cell_v1 = NR_Cell_v1(nr_sim=sim)
print(f'NR_Cell_v1 position is {nr_cell_v1.xyz}')   # Hooray! It works!

# Now what if we wanted to extend our NR_Cell to include a new parameter?
class NR_Cell_v2(Cell):
    def __init__(self, nr_sim_v2, magic_number):
        print(f"Today's magic number is {magic_number}!")
        super().__init__(sim=nr_sim_v2)

# Create a NR_Cell_v2 obj
nr_cell_v2 = NR_Cell_v2(nr_sim_v2=sim, magic_number=7)
print(f'NR_Cell_v2 position is {nr_cell_v2.xyz}')

# The Cell class has lots of keyword arguments we can set.
# How could we set the height of the cell?
class NR_Cell_v3(Cell):
    def __init__(self, nr_sim_v3, magic_number, nr_h_BS):
        print(f"Today's magic number is {magic_number}!")
        super().__init__(sim=nr_sim_v3, h_BS=nr_h_BS)

# Create a NR_Cell_v3 obj
nr_cell_v3 = NR_Cell_v3(nr_sim_v3=sim, magic_number=7, nr_h_BS=30)
print(f'NR_Cell_v3 position is {nr_cell_v3.xyz}')
# So then, can we access other Cell class parameters that haven't been set in the super().__init__?
print(f"This is the NR_Cell_v3 operating bandwidth: {nr_cell_v3.bw_MHz} MHz")
print(f"The simulator currently uses the centre frequency: {nr_cell_v3.sim.params['fc_GHz']} GHz.")


# Here comes the tricky bit...
# Let's use args and kwargs
#
# First in isolation....
#
#
#
# Define a function that prints the input argument
def my_one(arg1):
    print(arg1)

# Test it works
my_one('apple') # Great it prints 'apple' to the screen!

# Now what if we wanted to have more names added to that same function?
# my_one('apple', 'pear')
#
# We get a TypeError
#   `Exception has occurred: TypeError
#   my_one() takes 1 positional argument but 2 were given`


# To extend the function to ANY number of arguments, we use *args.
def my_one(*args):
    print(*args)

# Now we can add multiple args
my_one('apple','pear')

