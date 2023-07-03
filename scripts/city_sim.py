import datetime
import sys

import ses_ling.utils.sim as sim_utils

if __name__ == '__main__':
    s = float(sys.argv[1])
    q1 = float(sys.argv[2])
    q2 = float(sys.argv[3])
    cells_nr_users_th = int(sys.argv[4])
    nr_classes = int(sys.argv[5])
    nr_steps = int(sys.argv[6])

    focus_cities = [
        'London', 'Manchester', 'Birmingham', 'Liverpool', 'Leeds', 'Sheffield', 
        'Newcastle upon Tyne', 'Bristol, City of'
    ]
    for city in focus_cities:
        sim = sim_utils.Simulation.from_saved_state(
            city, s=s, q1=q1, q2=q2,
            cells_nr_users_th=cells_nr_users_th, nr_classes=nr_classes
        )
        sim.sim_id = f"_{datetime.datetime.now().isoformat()}"
        print(sim.to_dict())
        sim.run(nr_steps)
        sim.save_state()
