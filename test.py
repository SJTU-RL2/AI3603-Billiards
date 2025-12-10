import pooltool as pt

class TestMultiSystem:
    def __init__(self):
        self.table = pt.Table.default()
        self.balls = pt.get_rack(pt.GameType.EIGHTBALL, self.table)
        self.cue = pt.Cue(cue_ball_id="cue")

    def initialize(self):
        print("Running MultiSystem Test...")
        #print("Current table is:", self.table)
        print("Current balls are:", self.balls)
        shot = pt.System(table=self.table, balls=self.balls, cue=self.cue)
        shot.strike(phi=90)
        pt.simulate(shot, inplace=True)
        multisystem = pt.MultiSystem()
        multisystem.append(shot)
        next_shot = multisystem[-1].copy()
        next_shot.strike(phi=0)
        pt.simulate(next_shot, inplace=True)
        multisystem.append(next_shot)
        #pt.show(multisystem, title="MultiSystem Test")

def test_simulation():
    print("Running simulation test: simulate(system)")
    system = pt.System.example()
    simulated_system = pt.simulate(system)
    assert not system.simulated
    assert simulated_system.simulated
    print("Simulation test passed.")

    print("Running simulation test: simulate(system, inplace=True)")
    system2 = pt.System.example()
    simulated_system2 = pt.simulate(system2, inplace=True)
    assert system2.simulated
    assert simulated_system2.simulated
    assert system2 is simulated_system2
    print("In-place simulation test passed.")





if __name__ == "__main__":
    # test = TestMultiSystem()
    # test.initialize()
    test_simulation()