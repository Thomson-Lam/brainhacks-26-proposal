# BrainHacks 2026 


## Abstract 



## Introduction 


## System Architecture 



## Components 

Through this project, we aim to deliver a proof of concept using low budget EEG. We will

1. Build an EEG using the Ultracortex Mark III as the base, and a more budget friendly 8 channel PiEEG for the project.
2. Develop a data pipeline to sample and label brain signals + screen images and visual data.
3. Train an ensemble system to classify, predict and execute actions based on brain signals and prior desktop context.
4. Test and benchmark our system's effectiveness during usage.

The UltraCortex mark III was chosen over mark IV because the mark III proved to be more 3D printer friendly without the need for denser, injection moulded parts, and was perfect for a low budget use case.

# UltraCortex + PiEEG BoM

> **Frame sizing by head circumference:** Small = 50–55 cm | Medium = 55–60 cm | Large = 60–65 cm

| Part                             | Type        | Qty                  | Notes                                                                                                                                                                            | Link                                                                                                               | Price |
| -------------------------------- | ----------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----- |
| FRAME_FRONT                      | 3D-Printed  | x1                   | .STLs available in small / medium / large                                                                                                                                        |                                                                                                                    | Free  |
| FRAME_BACK                       | 3D-Printed  | x1                   | .STLs available in small / medium / large                                                                                                                                        |                                                                                                                    | Free  |
| OCTANUT                          | 3D-Printed  | x9                   | .STLs available in tight / normal / loose — start with normal; switch to loose if too tight for OCTABOLT, or tight if too loose (print the FRAME first, then print the OCTANUTs) |                                                                                                                    | Free  |
| OCTABOLT                         | 3D-Printed  | x9                   | For holding electrode holders.                                                                                                                                                   |                                                                                                                    | Free  |
| OCTARING                         | 3D-Printed  | x21<br>(x9 minimum)  | For holding Octabolts.                                                                                                                                                           |                                                                                                                    | Free  |
| ELECTRODE_HOLDER                 | 3D-Printed  | x21 <br>(x9 minimum) | For holding the electrodes (electrodes -> holders -> octabolt).                                                                                                                  |                                                                                                                    | Free  |
| QUADSTAR                         | 3D-Printed  | x21                  | for mounting octabolts in curved areas. NOTE: use a stretchy filament for this, like NinjaFlex or SemiFlex.                                                                      |                                                                                                                    | Free  |
| Comfy Insert                     | 3D-Printed  | x12 minimum          | For positions without electrodes.                                                                                                                                                |                                                                                                                    | Free  |
| OCTATOOL                         | 3D-Printed  | x1                   | Used for installing octabolts.                                                                                                                                                   |                                                                                                                    | Free  |
| Spring 1 (weak)                  | Hardware    | x6                   | Weak spring for mounting spikey (dry comb) electrodes                                                                                                                            |                                                                                                                    |       |
| Spring 2 (strong)                | Hardware    | x3                   | Strong spring for mounting non spikey electroedes                                                                                                                                |                                                                                                                    |       |
| Nuts (2-56 thread)               | Hardware    | x18                  | Anything equivalent to the link will work                                                                                                                                        |                                                                                                                    | TBD   |
| Bolts (2-56 thread)              | Hardware    | x9                   | Anything equivalent to the link will work                                                                                                                                        |                                                                                                                    | TBD   |
| Wiring                           | Hardware    | x11                  | Free (taken from Arduino Kit)                                                                                                                                                    | N/A                                                                                                                | N/A   |
| Dry spikey electrodes            | Electrode   | x6                   | 5 mm Ag/AgCl Comb Electrodes — for nodes with hair                                                                                                                               |                                                                                                                    |       |
| Dry non-spikey (flat) electrodes | Electrode   | x3                   | Disposable/Reusable Cup Wet/Dry EEG Electrode — for nodes without hair (e.g. forehead)                                                                                           |                                                                                                                    |       |
| Ear Clip Electrode               | Electrode   | x2                   | TDI-430 Silver-Silver Chloride Ear Clip Electrode — used as reference                                                                                                            |                                                                                                                    |       |
| **PiEEG Board**                  | Electronics | x1                   | PiEEG 8 channel board to replace Cyton                                                                                                                                           |                                                                                                                    |       |
| Raspberry Pi 4/5                 | Electronics | x1                   | Board to use with PiEEG                                                                                                                                                          | TBD                                                                                                                | TBD   |
| 5V USB Power Supply (UPS) for Pi | Electronics | x1                   | 5V max 2000mAh                                                                                                                                                                   | [link](https://www.sunfounder.com/products/sunfounder-raspberry-pi-4-ups-power-supply?_pos=1&_sid=b212fb1bf&_ss=r) |       |
| Zip Ties                         | Misc        | ~x10                 | Used for wiring and the quadstars, each quadstar needs 4 zipties.                                                                                                                | TBD                                                                                                                | TBD   |

> TODO: a verification/double checking is required for the electrodes and the components, for the spikey, non spikey, the springs for each, and the number of comfy inserts required for printing.

### Best Practices 


# Sources 

1. [Ultracortex electrodes](https://openbci.com/forum/index.php?p=/discussion/921/ultracortex-mark-iv-electrodes-units)
2. [Ultracortex Mark Iv Documentation](https://docs.openbci.com/AddOns/Headwear/MarkIV/#3d-printed-parts)
3. [Ultracortex Mark III Documentation](https://docs.openbci.com/Deprecated/MarkIII/https://docs.openbci.com/Deprecated/MarkIII/)
4. [Ultracortex Mark III Source Files](https://github.com/OpenBCI/Ultracortex/tree/master/Mark_III_Nova_REVISED)