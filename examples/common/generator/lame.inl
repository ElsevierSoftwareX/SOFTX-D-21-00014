/****************************************************************************
** 
**  Copyright (C) 2019-2020 Boris Krasnopolsky, Alexey Medvedev
**  Contact: xamg-test@imec.msu.ru
** 
**  This file is part of the XAMG library.
** 
**  Commercial License Usage
**  Licensees holding valid commercial XAMG licenses may use this file in
**  accordance with the terms of commercial license agreement.
**  The license terms and conditions are subject to mutual agreement
**  between Licensee and XAMG library authors signed by both parties
**  in a written form.
** 
**  GNU General Public License Usage
**  Alternatively, this file may be used under the terms of the GNU
**  General Public License, either version 3 of the License, or (at your
**  option) any later version. The license is as published by the Free 
**  Software Foundation and appearing in the file LICENSE.GPL3 included in
**  the packaging of this file. Please review the following information to
**  ensure the GNU General Public License requirements will be met:
**  https://www.gnu.org/licenses/gpl-3.0.html.
** 
****************************************************************************/

#pragma once

struct solid_block {
    size_t x1, x2, y1, y2, z1, z2;

  private:
    bool is_xwall(const size_t i) const {
        if ((i >= x1) && (i < x2))
            return true;
        else
            return false;
    }

    bool is_ywall(const size_t j) const {
        if ((j >= y1) && (j < y2))
            return true;
        else
            return false;
    }

    bool is_zwall(const size_t i) const {
        if ((i >= z1) && (i < z2))
            return true;
        else
            return false;
    }

  public:
    bool is_solid(const size_t i, const size_t j, const size_t k) {
        if (is_xwall(i) && is_ywall(j) && is_zwall(k))
            return true;
        else
            return false;
    }
};

struct grid_params {
    uint8_t scale;
    size_t nx, ny, nz;

    uint8_t order = 4;
    std::vector<double> xcoord_transform;
    std::vector<double> ycoord_transform;

    solid_block cube;

    grid_params(const uint8_t scale_) : scale(scale_) {
        xcoord_transform.resize(order, 0.0);
        ycoord_transform.resize(order, 0.0);

        if (scale == 1) {
            nx = 144;
            ny = 112;
            nz = 144;

            cube.x1 = 46;
            cube.x2 = 98;
            cube.y1 = 0;
            cube.y2 = 48;
            cube.z1 = 46;
            cube.z2 = 98;

            xcoord_transform[3] = 1.0 / 527436.0;
            xcoord_transform[2] = -216.0 / 527436.0;
            xcoord_transform[1] = 25019.0 / 527436.0;
            xcoord_transform[0] = -2.0;

            ycoord_transform[3] = 0.0;
            ycoord_transform[2] = 1.0 / 6720.0;
            ycoord_transform[1] = 92.0 / 6720.0;
            ycoord_transform[0] = 0.0;
        } else if (scale == 2) {
            nx = 240;
            ny = 168;
            nz = 240;

            cube.x1 = 70;
            cube.x2 = 170;
            cube.y1 = 0;
            cube.y2 = 74;
            cube.z1 = 70;
            cube.z2 = 170;

            xcoord_transform[3] = 1.0 / 1785000.0;
            xcoord_transform[2] = -360.0 / 1785000.0;
            xcoord_transform[1] = 58550.0 / 1785000.0;
            xcoord_transform[0] = -2.0;

            ycoord_transform[3] = 0.0;
            ycoord_transform[2] = 209.0 / 2921520.0;
            ycoord_transform[1] = 24014.0 / 2921520.0;
            ycoord_transform[0] = 0.0;
        } else if (scale == 3) {
            nx = 360;
            ny = 252;
            nz = 360;

            cube.x1 = 105;
            cube.x2 = 255;
            cube.y1 = 0;
            cube.y2 = 120;
            cube.z1 = 105;
            cube.z2 = 255;

            xcoord_transform[3] = 2.0 / 12048750.0;
            xcoord_transform[2] = -1080.0 / 12048750.0;
            xcoord_transform[1] = 263475.0 / 12048750.0;
            xcoord_transform[0] = -2.0;

            ycoord_transform[3] = 0.0;
            ycoord_transform[2] = 13.0 / 332640.0;
            ycoord_transform[1] = 1212.0 / 332640.0;
            ycoord_transform[0] = 0.0;
        } else {
            assert("Incorrect grid scale specified" && 0);
        }
    }

    void spline_xcoeff(std::vector<double> &a, double ic) {
        //  S(x) = a[3]*x^3 + a[2]*x^2 + a[1]*x + a[0]
        if (scale == 1) {
            if (ic < 14) {
                a[0] = 1.396162377508623;
                a[1] = 2.39973900862267;
                a[2] = .4213672254912018;
                a[3] = .03526915777851145;
            } else if (ic < 24) {
                a[0] = -3.307327090264011;
                a[1] = -7.600783718893908;
                a[2] = -6.666310657693695;
                a[3] = -1.639149241623937;
            } else if (ic < 46) {
                a[0] = -.2168199280894425;
                a[1] = 1.054198523421452;
                a[2] = 1.413131074030182;
                a[3] = .8749086296590155;
            } else if (ic < 56) {
                a[0] = .1785684516356735;
                a[1] = 3.426528801772148;
                a[2] = 6.157791630731574;
                a[3] = 4.038015667459944;
            } else if (ic < 88) {
                a[0] = 0.;
                a[1] = 1.610279383056608;
                a[2] = 0.;
                a[3] = -2.921086861223202;
            } else if (ic < 98) {
                a[0] = -.1785684516356735;
                a[1] = 3.426528801772148;
                a[2] = -6.157791630731574;
                a[3] = 4.038015667459944;
            } else if (ic < 120) {
                a[0] = .2168199280894425;
                a[1] = 1.054198523421452;
                a[2] = -1.413131074030183;
                a[3] = .8749086296590157;
            } else if (ic < 130) {
                a[0] = 3.307327090264012;
                a[1] = -7.600783718893908;
                a[2] = 6.666310657693695;
                a[3] = -1.639149241623937;
            } else {
                a[0] = -1.396162377508623;
                a[1] = 2.39973900862267;
                a[2] = -.4213672254912018;
                a[3] = .03526915777851145;
            }
        } else if (scale == 2) {
            if (ic < 9) {
                a[0] = 1.663364261477938;
                a[1] = 2.763116308992502;
                a[2] = .5920931128840505;
                a[3] = .06318801187864154;
            } else if (ic < 42) {
                a[0] = -1.114914960494785;
                a[1] = -2.080698526151822;
                a[2] = -2.222903782988859;
                a[3] = -.4821264969997728;
            } else if (ic < 70) {
                a[0] = -.4236400477834716;
                a[1] = .1334790192104974;
                a[2] = .1411254004985186;
                a[3] = .3592143418872756;
            } else if (ic < 90) {
                a[0] = .1851416166778065;
                a[1] = 3.786169005978167;
                a[2] = 7.446505374033857;
                a[3] = 5.229467657577501;
            } else if (ic < 150) {
                a[0] = 0.;
                a[1] = 1.752459555086567;
                a[2] = 0.;
                a[3] = -3.859087619448435;
            } else if (ic < 170) {
                a[0] = -.1851416166778064;
                a[1] = 3.786169005978166;
                a[2] = -7.446505374033855;
                a[3] = 5.229467657577499;
            } else if (ic < 198) {
                a[0] = .4236400477834715;
                a[1] = .1334790192104982;
                a[2] = -.1411254004985194;
                a[3] = .3592143418872759;
            } else if (ic < 231) {
                a[0] = 1.114914960494784;
                a[1] = -2.080698526151822;
                a[2] = 2.222903782988859;
                a[3] = -.4821264969997728;
            } else {
                a[0] = -1.663364261477938;
                a[1] = 2.763116308992505;
                a[2] = -.5920931128840512;
                a[3] = 0.0631880118786417;
            }
        } else if (scale == 3) {
            if (ic < 15) {
                a[0] = 1.031662508857209;
                a[1] = 1.969399191066359;
                a[2] = .2286523094234516;
                a[3] = 9.341705522873112e-4;
            } else if (ic < 65) {
                a[0] = -.7588674138101876;
                a[1] = -1.206056260460024;
                a[2] = -1.648541921732235;
                a[3] = -0.368971630917371;
            } else if (ic < 105) {
                a[0] = -.5215911974286817;
                a[1] = -.4253232006982945;
                a[2] = -.7922345627386506;
                a[3] = -.05590590211357671;
            } else if (ic < 132) {
                a[0] = .2395985844288585;
                a[1] = 4.141815490446947;
                a[2] = 8.342042819551832;
                a[3] = 6.033612352746744;
            } else if (ic < 228) {
                a[0] = 0.;
                a[1] = 1.693096447267712;
                a[2] = 0.;
                a[3] = -3.439323928921869;
            } else if (ic < 255) {
                a[0] = -.2395985844288585;
                a[1] = 4.141815490446947;
                a[2] = -8.342042819551834;
                a[3] = 6.033612352746745;
            } else if (ic < 295) {
                a[0] = .5215911974286818;
                a[1] = -.4253232006982945;
                a[2] = .7922345627386508;
                a[3] = -.05590590211357682;
            } else if (ic < 345) {
                a[0] = .7588674138101876;
                a[1] = -1.206056260460024;
                a[2] = 1.648541921732235;
                a[3] = -0.368971630917371;
            } else {
                a[0] = -1.03166250885721;
                a[1] = 1.969399191066359;
                a[2] = -0.228652309423452;
                a[3] = 9.341705522873667e-4;
            }
        }
    }

    void spline_ycoeff(std::vector<double> &a, double jc) {
        //  S(x) = a[3]*x^3 + a[2]*x^2 + a[1]*x + a[0]
        if (scale == 1) {
            if (jc < 10) {
                a[0] = 0.;
                a[1] = .4500000000000001;
                a[2] = -.2935533747052894;
                a[3] = 6.708531575913051;
            } else if (jc < 38) {
                a[0] = .03430441990176908;
                a[1] = -.2280167698232014;
                a[2] = 4.173380638247565;
                a[3] = -3.10120586429714;
            } else if (jc < 48) {
                a[0] = -2.563649298895133;
                a[1] = 10.37415873036821;
                a[2] = -10.24901194500876;
                a[3] = 3.438502513535689;
            } else if (jc < 64) {
                a[0] = -.008557016802147377;
                a[1] = 2.708881884089249;
                a[2] = -2.583735098729805;
                a[3] = .8834102314427034;
            } else if (jc < 87) {
                a[0] = 5.052109496470486;
                a[1] = -7.509771652326646;
                a[2] = 4.294204781550124;
                a[3] = -.6597173057995884;
            } else if (jc < 98) {
                a[0] = 7.504547694864687;
                a[1] = -10.6845712464721;
                a[2] = 5.664181874766423;
                a[3] = -0.856773023310955;
            } else {
                a[0] = -28.34061977576933;
                a[1] = 28.12523413526699;
                a[2] = -8.342364428417605;
                a[3] = .8282250281999057;
            }
        } else if (scale == 2) {
            if (jc < 20) {
                a[0] = 0.;
                a[1] = 0.5;
                a[2] = -.5684921832901438;
                a[3] = 5.517772964909718;
            } else if (jc < 59) {
                a[0] = .06233578376582055;
                a[1] = -.4689042295569277;
                a[2] = 4.451499809217404;
                a[3] = -3.151924491966832;
            } else if (jc < 74) {
                a[0] = -2.185198059852231;
                a[1] = 8.717375665526223;
                a[2] = -8.064105321035434;
                a[3] = 2.531927715361443;
            } else if (jc < 88) {
                a[0] = -2.21918495246137;
                a[1] = 8.81933634335364;
                a[2] = -8.166065998862852;
                a[3] = 2.565914607970583;
            } else if (jc < 120) {
                a[0] = 4.742597325172884;
                a[1] = -7.531522690118807;
                a[2] = 4.634803111498467;
                a[3] = -.7746285330177924;
            } else if (jc < 153) {
                a[0] = 1.635341021829459;
                a[1] = -2.908803297044359;
                a[2] = 2.342369691044022;
                a[3] = -.3956848463868132;
            } else {
                a[0] = -30.09333727085763;
                a[1] = 29.55298527669742;
                a[2] = -8.728242912540622;
                a[3] = .8628046865218802;
            }
        } else if (scale == 3) {
            if (jc < 20) {
                a[0] = 0.;
                a[1] = .8675000000000002;
                a[2] = -2.635398381455771;
                a[3] = 20.00101762126424;
            } else if (jc < 90) {
                a[0] = .01600837036955505;
                a[1] = .3248684456798112;
                a[2] = 3.495748365455494;
                a[3] = -3.090747028787434;
            } else if (jc < 120) {
                a[0] = -1.497835823305939;
                a[1] = 7.37167970490478;
                a[2] = -7.438346686641689;
                a[3] = 2.564502805042847;
            } else if (jc < 143) {
                a[0] = -0.257719998070995;
                a[1] = 3.651332229199951;
                a[2] = -3.717999210936858;
                a[3] = 1.324386979807903;
            } else if (jc < 180) {
                a[0] = 4.797985956621731;
                a[1] = -7.837124154581934;
                a[2] = 4.984025212216816;
                a[3] = -.8727526686501045;
            } else if (jc < 233) {
                a[0] = .7330112214661275;
                a[1] = -1.492467642278255;
                a[2] = 1.683089053788552;
                a[3] = -.3002930195533108;
            } else {
                a[0] = -28.27470371382332;
                a[1] = 27.801962358161;
                a[2] = -8.17823727456962;
                a[3] = 0.806239939703357;
            }
        }
    }

    void spline_zcoeff(std::vector<double> &a, double jc) { spline_xcoeff(a, jc); }
};

static inline double poly4(const std::vector<double> &a, const double x) {
    double val = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        val += a[i] * std::pow((x), i);
    return val;
}

struct cube_geometry {
    const double Lx = 4.0;
    const double Ly = 3.4;
    const double Lz = 4.0;
    double hx, hy, hz;

    grid_params grid;

    cube_geometry(const uint8_t scale_) : grid(scale_) {
        hx = Lx / grid.nx;
        hy = Ly / grid.ny;
        hz = Lz / grid.nz;
    }

    bool is_ext_ywall(const size_t j) const {
        if ((j < 0) || (j >= grid.ny))
            return true;
        else
            return false;
    }

    double transform_x1(double x) {
        double ic = x / hx;
        double xc = poly4(grid.xcoord_transform, ic);

        std::vector<double> a(4, 0.0);
        grid.spline_xcoeff(a, ic);

        return poly4(a, xc);
    }

    double transform_x2(double x) {
        double jc = x / hy;
        double yc = poly4(grid.ycoord_transform, jc);

        std::vector<double> a(4, 0.0);
        grid.spline_ycoeff(a, jc);

        return poly4(a, yc);
    }

    double transform_x3(double x) { return transform_x1(x); }

    double dtransform_x1(double x) {
        return ((transform_x1(x + 0.01 * hx) - transform_x1(x - 0.01 * hx)) / (0.02 * hx));
    }

    double dtransform_x2(double x) {
        return ((transform_x2(x + 0.01 * hy) - transform_x2(x - 0.01 * hy)) / (0.02 * hy));
    }

    double dtransform_x3(double x) {
        return ((transform_x3(x + 0.01 * hz) - transform_x3(x - 0.01 * hz)) / (0.02 * hz));
    }

    //-----------------------------------//

    double Lame_H1(double x1) { return dtransform_x1(x1); }

    double Lame_H2(double x2) { return dtransform_x2(x2); }

    double Lame_H3(double x3) { return dtransform_x3(x3); }

    double Lame_H(double x1, double x2, double x3) {
        return (Lame_H1(x1) * Lame_H2(x2) * Lame_H3(x3));
    }

    //------------------------------------------------------/

    double H_c(const size_t i, const size_t j, const size_t k) {
        double x1 = (i + 0.5) * hx;
        double x2 = (j + 0.5) * hy;
        double x3 = (k + 0.5) * hz;

        return Lame_H(x1, x2, x3);
    }

    double H_f1(const size_t i, const size_t j, const size_t k) {
        double x1 = i * hx;
        double x2 = (j + 0.5) * hy;
        double x3 = (k + 0.5) * hz;

        return Lame_H(x1, x2, x3);
    }

    double H_f2(const size_t i, const size_t j, const size_t k) {
        double x1 = (i + 0.5) * hx;
        double x2 = j * hy;
        double x3 = (k + 0.5) * hz;

        return Lame_H(x1, x2, x3);
    }

    double H_f3(const size_t i, const size_t j, const size_t k) {
        double x1 = (i + 0.5) * hx;
        double x2 = (j + 0.5) * hy;
        double x3 = k * hz;

        return Lame_H(x1, x2, x3);
    }

    double H1_f1(const size_t i) {
        double x1 = i * hx;
        return Lame_H1(x1);
    }

    double H2_f2(const size_t j) {
        double x2 = j * hy;
        return Lame_H2(x2);
    }

    double H3_f3(const size_t k) {
        double x3 = k * hz;
        return Lame_H3(x3);
    }
};
