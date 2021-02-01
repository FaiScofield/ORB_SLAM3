#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[])
{
    Vector3d euler_wc1(-M_PI_2, 0, -M_PI_2);	// yaw,pitch,roll
    Matrix3d Rbc1 = (AngleAxisd(euler_wc1[0], Vector3d::UnitZ()) *
                    AngleAxisd(euler_wc1[1], Vector3d::UnitY()) *
                    AngleAxisd(euler_wc1[2], Vector3d::UnitX())).matrix();
    cout << "Rbc1 = \n" << Rbc1 << endl;

    Isometry3d Tbc1 = Isometry3d::Identity();
    Tbc1.rotate(Rbc1);
    cout << "Tbc1 = \n" << Tbc1.matrix() << endl;

    Vector3d tc1c2(0, 0, -0.07);
    Matrix3d Rc1c2 = AngleAxisd(M_PI_2, Vector3d::UnitX()).matrix();
    cout << "Rc1c2 = \n" << Rc1c2 << endl;

    Isometry3d Tc1c2 = Isometry3d::Identity();
    Tc1c2.rotate(Rc1c2);
    Tc1c2.pretranslate(tc1c2);
    cout << "Tc1c2 = \n" << Tc1c2.matrix() << endl;

    Isometry3d Tbc2 = Tbc1 * Tc1c2;
    cout << "Tbc2 = \n" << Tbc2.matrix() << endl;


    return 0;
}
