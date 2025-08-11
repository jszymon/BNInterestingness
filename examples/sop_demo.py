import numpy

from BNInter import sop

# P(x0x1)
dist1 = ([0,1], numpy.array([[1.0/9,2.0/9,0.0/9],
                             [3.0/9,0.0/9,0.0/9],
                             [1.0/9,1.0/9,1.0/9]]))
# P(x2|x1)
dist2 = ([2,1], numpy.array([[1.0/3,0.0/3,0.0/3],
                             [1.0/3,2.0/3,1.0/3],
                             [1.0/3,1.0/3,2.0/3]]))


def do_demo():
    thesop = sop.simple_sop(3)
    thesop.add_factor(*dist1)
    thesop.add_factor(*dist2)

    thesop.prepare([0,0,0], [2,2,2])
    print(thesop)
    print("The sum is: %f" % thesop.compute())
    print("Complexity: O(|dom|^%d)" % thesop.complexity())
    print("Cost: %d adds, %d muls" % thesop.cost())
    print("Memory cost: %d" % thesop.mem_cost())
    print()

    thesop.prepare([0,0,0], [2,0,2])
    print(thesop)
    print("The sum is: %f" % thesop.compute())
    print("Complexity: O(|dom|^%d)" % thesop.complexity())
    print("Cost: %d adds, %d muls" % thesop.cost())
    print("Memory cost: %d" % thesop.mem_cost())
    print()

    thesop.prepare([0,1,0], [2,1,2])
    print("The sum is: %f" % thesop.compute())
    print("Complexity: O(|dom|^%d)" % thesop.complexity())
    print("Cost: %d adds, %d muls" % thesop.cost())
    print("Memory cost: %d" % thesop.mem_cost())
    print()

    thesop.prepare([0,2,0], [2,2,2])
    print("The sum is: %f" % thesop.compute())
    print("Complexity: O(|dom|^%d)" % thesop.complexity())
    print("Cost: %d adds, %d muls" % thesop.cost())
    print("Memory cost: %d" % thesop.mem_cost())
    print()

    asop = sop.array_sop(3)
    asop.add_factor(*dist1)
    asop.add_factor(*dist2)
    asop.prepare([1])
    print(asop)
    print("The sum is: " + str(asop.compute()))
    print("Complexity: O(|dom|^%d)" % asop.complexity())
    print("Cost: %d adds, %d muls" % asop.cost())
    print("Memory cost: %d" % asop.mem_cost())
    print() 
    asop.prepare([])
    print(asop)
    print("The sum is: " + str(asop.compute()))
    print("Complexity: O(|dom|^%d)" % asop.complexity())
    print("Cost: %d adds, %d muls" % asop.cost())
    print("Memory cost: %d" % asop.mem_cost())
    print()
    asop.prepare([0,1,2])
    print(asop)
    print("The sum is: " + str(asop.compute()))
    print("Complexity: O(|dom|^%d)" % asop.complexity())
    print("Cost: %d adds, %d muls" % asop.cost())
    print("Memory cost: %d" % asop.mem_cost())
    print()
    asop.prepare([0,1])
    print(asop)
    print("The sum is: " + str(asop.compute()))
    print("Complexity: O(|dom|^%d)" % asop.complexity())
    print("Cost: %d adds, %d muls" % asop.cost())
    print("Memory cost: %d" % asop.mem_cost())
    print()
    asop.prepare([1,0])
    print(asop)
    print("The sum is: " + str(asop.compute()))
    print("Complexity: O(|dom|^%d)" % asop.complexity())
    print("Cost: %d adds, %d muls" % asop.cost())
    print("Memory cost: %d" % asop.mem_cost())
    print()

if __name__ == '__main__':
    do_demo()

