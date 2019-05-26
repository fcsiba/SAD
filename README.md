# SAD
FYP Fall 2018:

This project lies on the intersection between Network Security and Machine Learning. Most Intrusion Detection Systems (IDS) today, work on signature based detection which results in new variants of network attacks being undetected until their signatures are updated into the system. IDS like these cannot stop a network attack whose signature is not in its database. To rectify this, instead of relying on a database for signatures, we have created a machine learning model that can classify new variants of attacks instantly. This is done by training the model on a number of classes of attacks, after which it can, detect new variants of attacks within those classes. 

The model here, is trained on Portscan, BruteForceSSH, Cross-site Scripting, SQL Injection, and Denial of Service attacks. New variations within these attacks are successfully detected by the model with an accuracy of 0.96%. These classes of attacks were chosen as they are one of the most common types of attack as according to OWASP, symantec and numerous other surveys and [reports](https://www.symantec.com/security-center/threat-report). 

Both training data and test data was generated in the Robotics Lab at IBA City campus, the attacks were carried out in an isolated lab environment that we set up. [Wireshark](https://www.wireshark.org) was used to capture this data into pcap files through a SPAN port on the Cisco switch. These are converted to CSV files by [CICFlowMeter](https://www.unb.ca/cic/research/applications.html#CICFlowMeter), feature extraction from the pcaps is also done here. 

Variations within each attack were confirmed and tested by running them against [Snort IDS/IPS](https://www.snort.org). Each attack has atleast one or more variatons that could not be detected by Snort but can be detected by our model. 

This model has been integrated with [TCPDump](https://www.tcpdump.org) and [CICFlowMeter](https://www.unb.ca/cic/research/applications.html#CICFlowMeter) so that the finished product can capture traffic, convert the captured traffic from Pcap to CSV, feed it into the model and classify the captured traffic as malicious or benign.   

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. To run the application, please 

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [TCPDump](https://www.tcpdump.org) - The portable C/C++ library used for network traffic capture.
* [CICFlowMeter](https://www.unb.ca/cic/research/applications.html#CICFlowMeter) - The network traffic flow generator and analyser.
* [Sci-kit learn](https://scikit-learn.org/stable/) -  Python tool for data mining and data analysis, built on NumPy, SciPy, and matplotlib libraries.


## Authors

* **Syeda Mahnur Asif** - [MahnurA](https://github.com/PurpleBooth)
* **Sunila Aftab** - [sunila-aftab](https://github.com/sunila-aftab)
* **Emaan Hasan** - [emaan-hasan](https://github.com/emaan-hasan)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details




