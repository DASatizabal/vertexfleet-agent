# 03 Logic: First-Order Ontology

## Classes (unary predicates)

| Class | Instances | Notes |
|---|---|---|
| Vehicle(x) | V01 to V50 | Has age, odometer, telemetry |
| Zone(x) | Z1 to Z8 | Demand is forecast per zone and hour |
| Depot(x) | D1 to D3 | Each has service bays |
| ServiceBay(x) | one per depot per shift | Capacity 1 vehicle per shift |
| Technician(x) | one per depot per shift | Required for any service |
| Trip(x) | generated hourly | Requested in a zone |

## Relations

| Relation | Signature | Meaning |
|---|---|---|
| LocatedIn(v, z) | Vehicle x Zone | Vehicle v is in zone z |
| Adjacent(z, y) | Zone x Zone | Zones share a border (symmetric) |
| HasBay(d, b) | Depot x ServiceBay | Bay b belongs to depot d |
| AssignedTo(v, d) | Vehicle x Depot | Vehicle v is scheduled at depot d |
| ServicedBy(v, t) | Vehicle x Technician | Technician t works on vehicle v |
| RequestedIn(r, z) | Trip x Zone | Trip r starts in zone z |

## Axioms

1. Every vehicle is in exactly one zone:
   FORALL v ( Vehicle(v) => EXISTS z ( Zone(z) AND LocatedIn(v, z) AND FORALL y ( LocatedIn(v, y) => y = z ) ) )
2. Adjacency is symmetric:
   FORALL z, y ( Adjacent(z, y) => Adjacent(y, z) )
3. A vehicle in service cannot be dispatched (rule R7):
   FORALL v ( InService(v) => NOT AvailableForDispatch(v) )
4. A bay holds at most one vehicle per shift (the CSP constraint from 02_simulation):
   FORALL v1, v2, b, s ( AssignedBay(v1, b, s) AND AssignedBay(v2, b, s) => v1 = v2 )
5. Inspection needs evidence (rules R1 and R2):
   FORALL v ( ( OverdueService(v) AND HighVibration(v) ) OR EngineHot(v) => NeedsInspection(v) )
6. Reposition only toward a surge from a quieter neighbor (rule R5):
   FORALL z, y ( DemandSurge(z) AND Adjacent(z, y) AND LowDemand(y) => RepositionCandidate(y, z) )

## From FOL to code

The knowledge base works with ground atoms (constants filled in) and Horn rules with variables. Axioms 3, 5 and 6 are implemented directly as rules R7, R1/R2 and R5. Axiom 4 is enforced by the CSP scheduler, not by forward chaining. Axioms 1 and 2 are guaranteed by how the simulator builds the map and positions.
